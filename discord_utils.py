import re
import discord
from typing import Dict, Any

async def translate_user_mentions(text: str, guild: discord.Guild) -> str:
    """
    Convert <@ID> or <@!ID> mentions in text to @DisplayName.
    """
    if not guild or not text:
        return text
    
    mention_pattern = re.compile(r'<@!?(\d+)>')
    
    # We might need to fetch members if they aren't in cache
    # But for a bot, usually we use get_member
    
    def replace_mention(match):
        user_id = int(match.group(1))
        member = guild.get_member(user_id)
        if member:
            return f"@{member.display_name}"
        return match.group(0)
    
    return mention_pattern.sub(replace_mention, text)

async def convert_usernames_to_mentions(text: str, guild: discord.Guild) -> str:
    """
    Convert @DisplayName in text to <@ID> mentions.
    Matches against guild members' display names.
    """
    if not guild or not text:
        return text
        
    # Get all members and sort by display name length descending to match longest first
    # This prevents partial matches (e.g., matching @Alice in @AliceSmith)
    members = sorted(guild.members, key=lambda m: len(m.display_name), reverse=True)
    
    result = text
    for member in members:
        display_name = member.display_name
        # Skip very short names or names that might cause too many false positives
        if len(display_name) < 2:
            continue
            
        # Use regex to match @DisplayName with word boundaries
        # We escape the display name to handle special regex characters
        escaped_name = re.escape(display_name)
        # Match @ followed by the display name, ensuring it's not followed by more word characters
        pattern = re.compile(f'@{escaped_name}(?!\\w)')
        result = pattern.sub(member.mention, result)
        
    return result

def format_metadata(token_stats: Dict[str, Any], duration: float, tool_counts: Dict[str, int]) -> str:
    """
    Format query execution metadata for inclusion in the response.
    """
    if not token_stats:
        return f"\n\n-# _Ref: {duration:.1f}s_"
        
    stats_parts = []
    
    for model, stats in token_stats.items():
        m_in = stats.get('total_input_tokens', 0)
        m_out = stats.get('total_output_tokens', 0)
        # Shorten model name for display (e.g., mistralai/mistral-7b-instruct -> mistral-7b-instruct)
        short_model = model.split('/')[-1] if '/' in model else model
        stats_parts.append(f"{short_model}: {m_in}+{m_out}")
    
    tools_str = ""
    if tool_counts:
        # Filter out 0 counts and sort by count descending
        active_tools = sorted([(n, c) for n, c in tool_counts.items() if c > 0], key=lambda x: x[1], reverse=True)
        if active_tools:
            tools_list = [f"{name}({count})" for name, count in active_tools]
            tools_str = " | Tools: " + ", ".join(tools_list)
        
    return f"\n\n-# *{' | '.join(stats_parts)} | {duration:.1f}s{tools_str}*"
