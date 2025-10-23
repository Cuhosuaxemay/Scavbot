import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional
import discord
from discord import app_commands
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import os
import yaml
import sys

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- DATA PATHS ---
DATA_DIR = "/app/data"
CONFIG_FILE = os.path.join(DATA_DIR, "config.yaml")
ROLE_STATE_FILE = os.path.join(DATA_DIR, "role_states.yaml")

# --- ROLE ID CONSTANTS ---
MODERATOR_ROLE_IDS = [1404958985882173480, 1093445724177432646]
C_SUITE_ROLE_ID = 1093445724177432646

def has_any_role_id(role_ids):
    def predicate(interaction):
        return any(role.id in role_ids for role in getattr(interaction.user, "roles", []))
    return app_commands.check(predicate)

# --- DATA HANDLING FUNCTIONS ---
def load_role_states():
    if not os.path.exists(ROLE_STATE_FILE):
        return {}
    try:
        with open(ROLE_STATE_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, IOError):
        logging.exception("Error loading role_states.yaml")
        return {}

def save_role_states(states):
    try:
        with open(ROLE_STATE_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(states, f)
    except (yaml.YAMLError, IOError):
        logging.exception("Error saving role_states.yaml")

def get_config(filename: str = CONFIG_FILE) -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)

# --- CONSTANTS ---
VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
MAX_MESSAGE_NODES = 500

# --- BOT SETUP ---
config = get_config()
curr_model = next(iter(config["models"]))
msg_nodes = {}
last_task_time = 0
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)
httpx_client = httpx.AsyncClient()

@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

# --- HELPER FUNCTIONS ---
async def enforce_role_state(member: discord.Member):
    role_states = load_role_states()
    user_id = str(member.id)
    state = role_states.get(user_id)
    if not state: return
    guild = member.guild
    special_role_name = None
    if state.get("type") == "shadowban": special_role_name = "shadowbanned"
    elif state.get("type") == "ghost": special_role_name = "ghosted"
    else: return
    special_role = discord.utils.get(guild.roles, name=special_role_name)
    if not special_role: return
    desired_roles = [role for role in member.roles if role.is_default()] + [special_role]
    current_role_ids = {role.id for role in member.roles}
    desired_role_ids = {role.id for role in desired_roles}
    if current_role_ids != desired_role_ids:
        try:
            await member.edit(roles=desired_roles, reason=f"Persistent {state['type']} enforcement")
        except (discord.Forbidden, discord.HTTPException): pass

async def send_mod_announcement(interaction: discord.Interaction, action: str, target_member: discord.Member):
    c_suite_role_ping = f"<@&{C_SUITE_ROLE_ID}>"
    announcement = f"{c_suite_role_ping} **Moderation Action Logged**\n- **Moderator:** {interaction.user.mention}\n- **Action:** {action}\n- **Target:** {target_member.mention}"
    try:
        await interaction.channel.send(announcement)
    except (discord.Forbidden, discord.HTTPException) as e:
        logging.error(f"Could not send moderation announcement in channel {interaction.channel.id}: {e}")

# --- COMMANDS ---
@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model
    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."
    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))

@discord_bot.tree.command(name="shadowban", description="Remove all roles and assign the shadowbanned role to a user (persistent)")
@has_any_role_id(MODERATOR_ROLE_IDS)
async def shadowban_command(interaction: discord.Interaction, member: discord.Member):
    SHADOWBAN_ROLE_NAME = "shadowbanned"
    await interaction.response.defer(ephemeral=True)
    role_states = load_role_states()
    user_id = str(member.id)
    guild = interaction.guild
    shadowban_role = discord.utils.get(guild.roles, name=SHADOWBAN_ROLE_NAME)
    if not shadowban_role:
        shadowban_role = await guild.create_role(name=SHADOWBAN_ROLE_NAME, reason="Shadowban command issued")
    if user_id not in role_states or role_states[user_id].get("type") != "shadowban":
        original_roles = [role.id for role in member.roles if not role.is_default() and role != shadowban_role]
        role_states[user_id] = {"type": "shadowban", "roles": original_roles}
        save_role_states(role_states)
        logging.info(f"Saved original roles for {member.name}: {original_roles}")
    await enforce_role_state(member)
    await interaction.followup.send(f"{member.mention} has been shadowbanned.", ephemeral=True)
    await send_mod_announcement(interaction, "Shadowban", member)

@discord_bot.tree.command(name="unshadow", description="Restore roles to a previously shadowbanned user")
@has_any_role_id(MODERATOR_ROLE_IDS)
async def unshadow_command(interaction: discord.Interaction, member: discord.Member):
    await interaction.response.defer(ephemeral=True)
    role_states = load_role_states()
    user_id = str(member.id)

    if user_id in role_states and role_states[user_id].get("type") == "shadowban":
        original_role_ids = role_states[user_id].get("roles", [])
        logging.info(f"Found saved role IDs for {member.name}: {original_role_ids}")

        final_roles = []
        bot_top_role = interaction.guild.me.top_role

        for role_id in original_role_ids:
            role = member.guild.get_role(role_id)
            if role:
                # --- NEW HIERARCHY CHECK ---
                if role >= bot_top_role:
                    logging.error(f"Cannot restore role '{role.name}' for {member.name} because it is higher than my top role.")
                    await interaction.followup.send(f":warning: I cannot restore the role `{role.name}` because it is higher than my role in the server's role list. Please adjust the role hierarchy.", ephemeral=True)
                    return
                final_roles.append(role)
            else:
                logging.warning(f"Could not find role with ID {role_id} to restore for {member.name}.")

        logging.info(f"Attempting to restore {member.name} with roles: {[r.name for r in final_roles]}")

        # Remove user from role_states BEFORE restoring roles to prevent enforcement race
        del role_states[user_id]
        save_role_states(role_states)

        try:
            await member.edit(roles=final_roles, reason="Unshadowed")
        except discord.Forbidden:
            logging.error("Forbidden: I lack the 'Manage Roles' permission.")
            await interaction.followup.send("I don't have the `Manage Roles` permission to perform this action.", ephemeral=True)
            return
        except discord.HTTPException as e:
            logging.error(f"HTTPException while editing roles: {e}")
            await interaction.followup.send(f"An error occurred while restoring roles: {e}", ephemeral=True)
            return

        await interaction.followup.send(f"{member.mention} has been unshadowed and their roles have been restored.", ephemeral=True)
        await send_mod_announcement(interaction, "Unshadow", member)
    else:
        await interaction.followup.send(f"{member.mention} is not currently shadowbanned.", ephemeral=True)

@discord_bot.tree.command(name="ghost", description="Remove all roles and assign the ghosted role to a user (persistent)")
@has_any_role_id(MODERATOR_ROLE_IDS)
async def ghost_command(interaction: discord.Interaction, member: discord.Member):
    GHOST_ROLE_NAME = "ghosted"
    await interaction.response.defer(ephemeral=True)
    role_states = load_role_states()
    user_id = str(member.id)
    guild = interaction.guild
    ghost_role = discord.utils.get(guild.roles, name=GHOST_ROLE_NAME)
    if not ghost_role:
        ghost_role = await guild.create_role(name=GHOST_ROLE_NAME, reason="Ghost command issued")
    if user_id not in role_states or role_states[user_id].get("type") != "ghost":
        original_roles = [role.id for role in member.roles if not role.is_default() and role != ghost_role]
        role_states[user_id] = {"type": "ghost", "roles": original_roles}
        save_role_states(role_states)
        logging.info(f"Saved original roles for {member.name}: {original_roles}")
    await enforce_role_state(member)
    await interaction.followup.send(f"{member.mention} has been ghosted.", ephemeral=True)
    await send_mod_announcement(interaction, "Ghost", member)

@discord_bot.tree.command(name="unghost", description="Restore roles to a previously ghosted user")
@has_any_role_id(MODERATOR_ROLE_IDS)
async def unghost_command(interaction: discord.Interaction, member: discord.Member):
    await interaction.response.defer(ephemeral=True)
    role_states = load_role_states()
    user_id = str(member.id)
    
    if user_id in role_states and role_states[user_id].get("type") == "ghost":
        original_role_ids = role_states[user_id].get("roles", [])
        logging.info(f"Found saved role IDs for {member.name}: {original_role_ids}")

        final_roles = []
        bot_top_role = interaction.guild.me.top_role

        for role_id in original_role_ids:
            role = member.guild.get_role(role_id)
            if role:
                # --- NEW HIERARCHY CHECK ---
                if role >= bot_top_role:
                    logging.error(f"Cannot restore role '{role.name}' for {member.name} because it is higher than my top role.")
                    await interaction.followup.send(f":warning: I cannot restore the role `{role.name}` because it is higher than my role in the server's role list. Please adjust the role hierarchy.", ephemeral=True)
                    return
                final_roles.append(role)
            else:
                logging.warning(f"Could not find role with ID {role_id} to restore for {member.name}.")

        logging.info(f"Attempting to restore {member.name} with roles: {[r.name for r in final_roles]}")
        
        # Remove user from role_states BEFORE restoring roles to prevent enforcement race
        del role_states[user_id]
        save_role_states(role_states)

        try:
            await member.edit(roles=final_roles, reason="Unghosted")
        except discord.Forbidden:
            logging.error("Forbidden: I lack the 'Manage Roles' permission.")
            await interaction.followup.send("I don't have the `Manage Roles` permission to perform this action.", ephemeral=True)
            return
        except discord.HTTPException as e:
            logging.error(f"HTTPException while editing roles: {e}")
            await interaction.followup.send(f"An error occurred while restoring roles: {e}", ephemeral=True)
            return

        await interaction.followup.send(f"{member.mention} has been unghosted and their roles have been restored.", ephemeral=True)
        await send_mod_announcement(interaction, "Unghost", member)
    else:
        await interaction.followup.send(f"{member.mention} is not currently ghosted.", ephemeral=True)

@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config
    if curr_str == "":
        config = await asyncio.to_thread(get_config)
    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]
    return choices[:25]

# --- EVENT HANDLERS AND MAIN LOOP ---
# ... (The rest of your code is unchanged and can be kept as is)
@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")
    await discord_bot.tree.sync()
    logging.info(f'Logged in as {discord_bot.user}')

@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time
    is_dm = new_msg.channel.type == discord.ChannelType.private
    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return
    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))
    config = await asyncio.to_thread(get_config)
    allow_dms = config.get("allow_dms", True)
    permissions = config["permissions"]
    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]
    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )
    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)
    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)
    if is_bad_user or is_bad_channel:
        return
    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
    provider_config = config["providers"][provider]
    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    model_parameters = config["models"].get(provider_slash_model, None)
    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None
    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)
    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)
    messages, user_warnings = [], set()
    curr_msg = new_msg
    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())
        async with curr_node.lock:
            if curr_node.text is None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()
                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])
                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, (embed.footer.text if embed.footer else '')))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if isinstance(component, TextDisplay)]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )
                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("image")
                ]
                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"
                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)
                try:
                    if curr_msg.reference is None and discord_bot.user.mention not in curr_msg.content:
                        history = [m async for m in curr_msg.channel.history(before=curr_msg, limit=1)]
                        prev_msg_in_channel = history[0] if history else None
                        if prev_msg_in_channel and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply):
                            is_private_convo = curr_msg.channel.type == discord.ChannelType.private
                            is_bot_turn = prev_msg_in_channel.author == discord_bot.user
                            is_user_turn = prev_msg_in_channel.author == curr_msg.author
                            if (is_private_convo and is_bot_turn) or (not is_private_convo and is_user_turn):
                                curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference is None and curr_msg.channel.parent.type == discord.ChannelType.text
                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)
                except (discord.NotFound, discord.HTTPException):
                    curr_node.fetch_parent_failed = True
            
            content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images] if curr_node.images else curr_node.text[:max_text]
            if content:
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id is not None:
                    message["name"] = str(curr_node.user_id)
                messages.append(message)
            if len(curr_node.text or "") > max_text: user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images: user_warnings.add(f"⚠️ Max {max_images} image(s) per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments: user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg is not None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message(s)")
            curr_msg = curr_node.parent_msg
    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."
        messages.append(dict(role="system", content=system_prompt))
    
    curr_content, finish_reason = None, None
    response_msgs, response_contents = [], []
    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)
    
    use_plain_responses = config.get("use_plain_responses", False)
    max_message_length = 4000 if use_plain_responses else 4096 - len(STREAMING_INDICATOR)
    embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)])) if not use_plain_responses else None
    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)
        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()
    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason is not None: break
                if not (choice := chunk.choices[0] if chunk.choices else None): continue
                finish_reason = choice.finish_reason
                delta_content = choice.delta.content or ""
                
                if not response_contents and not delta_content: continue
                
                if not response_contents or len(response_contents[-1] + delta_content) > max_message_length:
                    response_contents.append("")
                
                response_contents[-1] += delta_content
                
                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time
                    if time_delta >= EDIT_DELAY_SECONDS or finish_reason is not None:
                        is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")
                        embed.description = response_contents[-1] if finish_reason is not None else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if is_good_finish else EMBED_COLOR_INCOMPLETE
                        
                        if len(response_msgs) < len(response_contents):
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await response_msgs[-1].edit(embed=embed)
                        last_task_time = datetime.now().timestamp()
            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))
    except Exception:
        logging.exception("Error while generating response")
    
    full_response_text = "".join(response_contents)
    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = full_response_text
        msg_nodes[response_msg.id].lock.release()
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)
@discord_bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    await enforce_role_state(after)
@discord_bot.event
async def on_member_join(member: discord.Member):
    await enforce_role_state(member)
async def main() -> None:
    await discord_bot.start(config["bot_token"])
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass