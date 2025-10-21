import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional
import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import os
import yaml
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- DATA PATHS ---
DATA_DIR = "/app/data"
CONFIG_FILE = os.path.join(DATA_DIR, "config.yaml")
ROLE_STATE_FILE = os.path.join(DATA_DIR, "role_states.yaml")

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

# --- INTENTS FIX ---
# Enable the required 'members' intent for on_member_join and on_member_update
intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # <--- THIS IS THE CRITICAL FIX

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

# --- HELPER FUNCTION FOR ROLE ENFORCEMENT ---
async def enforce_role_state(member: discord.Member):
    """A helper function to enforce shadowban/ghost states on a member."""
    role_states = load_role_states()
    user_id = str(member.id)
    state = role_states.get(user_id)

    if not state:
        return # Do nothing if the user is not in our records

    guild = member.guild
    special_role_name = None
    if state.get("type") == "shadowban":
        special_role_name = "shadowbanned"
    elif state.get("type") == "ghost":
        special_role_name = "ghosted"
    else:
        return # Unknown state type

    # Ensure the special role exists
    special_role = discord.utils.get(guild.roles, name=special_role_name)
    if not special_role:
        logging.warning(f"'{special_role_name}' role not found, cannot enforce state for {member.name}.")
        return

    # Use member.edit for a single, efficient API call
    desired_roles = [role for role in member.roles if role.is_default()] + [special_role]
    current_role_ids = {role.id for role in member.roles}
    desired_role_ids = {role.id for role in desired_roles}

    # Only edit the member if their roles are not what we want them to be
    if current_role_ids != desired_role_ids:
        try:
            await member.edit(roles=desired_roles, reason=f"Persistent {state['type']} enforcement")
            logging.info(f"Enforced {state['type']} state for {member.name}.")
        except discord.Forbidden:
            logging.error(f"Missing permissions to enforce roles for {member.name}.")
        except discord.HTTPException as e:
            logging.error(f"Failed to enforce roles for {member.name}: {e}")

# --- COMMANDS ---
@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    # (Your model command code remains the same)
    global curr_model
    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."
    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))

@discord_bot.tree.command(name="shadowban", description="Remove all roles and assign the shadowbanned role to a user (persistent)")
@commands.has_permissions(administrator=True)
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

    await enforce_role_state(member) # Use the helper to apply the roles
    await interaction.followup.send(f"{member.mention} has been shadowbanned.", ephemeral=True)

@discord_bot.tree.command(name="unshadow", description="Restore roles to a previously shadowbanned user")
@commands.has_permissions(administrator=True)
async def unshadow_command(interaction: discord.Interaction, member: discord.Member):
    # (Your unshadow code remains mostly the same, just simplified)
    await interaction.response.defer(ephemeral=True)
    role_states = load_role_states()
    user_id = str(member.id)
    
    if user_id in role_states and role_states[user_id].get("type") == "shadowban":
        original_role_ids = role_states.get(user_id, {}).get("roles", [])
        roles_to_add = [member.guild.get_role(rid) for rid in original_role_ids if member.guild.get_role(rid)]
        
        # Remove the shadowban role
        shadowban_role = discord.utils.get(member.guild.roles, name="shadowbanned")
        if shadowban_role:
            roles_to_add = [r for r in roles_to_add if r.id != shadowban_role.id]
        
        await member.edit(roles=roles_to_add, reason="Unshadowed")
        
        del role_states[user_id]
        save_role_states(role_states)
        await interaction.followup.send(f"{member.mention} has been unshadowed.", ephemeral=True)
    else:
        await interaction.followup.send(f"{member.mention} is not currently shadowbanned.", ephemeral=True)


@discord_bot.tree.command(name="ghost", description="Remove all roles and assign the ghosted role to a user (persistent)")
@commands.has_permissions(administrator=True)
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

    await enforce_role_state(member) # Use the helper to apply the roles
    await interaction.followup.send(f"{member.mention} has been ghosted.", ephemeral=True)

@discord_bot.tree.command(name="unghost", description="Restore roles to a previously ghosted user")
@commands.has_permissions(administrator=True)
async def unghost_command(interaction: discord.Interaction, member: discord.Member):
    # (Your unghost code remains mostly the same, just simplified)
    await interaction.response.defer(ephemeral=True)
    role_states = load_role_states()
    user_id = str(member.id)
    
    if user_id in role_states and role_states[user_id].get("type") == "ghost":
        original_role_ids = role_states.get(user_id, {}).get("roles", [])
        roles_to_add = [member.guild.get_role(rid) for rid in original_role_ids if member.guild.get_role(rid)]

        # Remove the ghost role
        ghost_role = discord.utils.get(member.guild.roles, name="ghosted")
        if ghost_role:
            roles_to_add = [r for r in roles_to_add if r.id != ghost_role.id]
        
        await member.edit(roles=roles_to_add, reason="Unghosted")

        del role_states[user_id]
        save_role_states(role_states)
        await interaction.followup.send(f"{member.mention} has been unghosted.", ephemeral=True)
    else:
        await interaction.followup.send(f"{member.mention} is not currently ghosted.", ephemeral=True)


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    # (Your autocomplete code remains the same)
    global config
    if curr_str == "":
        config = await asyncio.to_thread(get_config)
    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]
    return choices[:25]

# --- EVENT HANDLERS ---
@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")
    await discord_bot.tree.sync()
    logging.info(f'Logged in as {discord_bot.user}')

@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    # (Your extensive on_message logic remains the same)
    # NOTE: I have fixed two syntax errors that were in your original code
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
    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg
    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())
        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()
                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])
                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )
                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]
                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"
                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)
                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text
                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)
                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True
            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]
            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)
                messages.append(message)
            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")
            curr_msg = curr_node.parent_msg
    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")
    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."
        messages.append(dict(role="system", content=system_prompt))
    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []
    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)
    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))
    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)
        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()
    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason != None:
                    break
                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue
                finish_reason = choice.finish_reason
                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""
                new_content = prev_content if finish_reason == None else (prev_content + curr_content)
                if response_contents == [] and new_content == "":
                    continue
                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")
                response_contents[-1] += new_content
                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time
                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")
                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE
                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)
                        last_task_time = datetime.now().timestamp()
            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))
    except Exception:
        logging.exception("Error while generating response")
    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()
    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)

@discord_bot.event
async def on_member_update(before: discord.Member, after: discord.Member):
    # This event is now simpler, just calling the helper function
    await enforce_role_state(after)

@discord_bot.event
async def on_member_join(member: discord.Member):
    # This event now also calls the same helper function
    await enforce_role_state(member)

async def main() -> None:
    await discord_bot.start(config["bot_token"])

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
