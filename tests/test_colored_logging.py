#!/usr/bin/env python3
"""
Test script for colored logging and reaction-based evaluation logging.
This script verifies that the colored output is working and reaction handlers are properly set up.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
from datetime import datetime
from discord_bot import DiscordBot
from colorama import Fore, Style


def test_colorama_import():
    """Test that colorama is properly imported (now via utils module)."""
    print("Testing colorama imports...")
    print("="*60)
    
    # Check utils.py has colorama
    utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils.py')
    with open(utils_path, 'r') as f:
        content = f.read()
    
    assert 'from colorama import Fore, Style, init' in content, \
        "colorama should be imported in utils.py"
    assert 'init(autoreset=True)' in content, \
        "colorama should be initialized in utils.py"
    
    print("✓ colorama properly imported and initialized in utils.py")
    
    # Check discord_bot.py imports from utils
    discord_bot_path = os.path.join(os.path.dirname(__file__), '..', 'discord_bot.py')
    with open(discord_bot_path, 'r') as f:
        content = f.read()
    
    assert 'from colorama import Fore, Style' in content, \
        "colorama Fore and Style should still be imported for colored output in discord_bot.py"
    assert 'from utils import setup_logging, CHARS_PER_TOKEN' in content, \
        "utils should be imported in discord_bot.py"
    
    print("✓ discord_bot.py imports colorama and utils correctly")
    
    # Check agent.py imports from utils
    agent_path = os.path.join(os.path.dirname(__file__), '..', 'agent.py')
    with open(agent_path, 'r') as f:
        content = f.read()
    
    assert 'from colorama import Fore, Style' in content, \
        "colorama Fore and Style should still be imported for colored output in agent.py"
    assert 'from utils import setup_logging, CHARS_PER_TOKEN' in content, \
        "utils should be imported in agent.py"
    
    print("✓ agent.py imports colorama and utils correctly")
    
    print("\n" + "="*60)
    print("✓ Colorama import test passed!")
    print("="*60)


def test_colored_logging_exists():
    """Test that colored logging statements exist in the code."""
    print("\nTesting colored logging implementation...")
    print("="*60)
    
    # Check discord_bot.py for colored user queries (green)
    discord_bot_path = os.path.join(os.path.dirname(__file__), '..', 'discord_bot.py')
    with open(discord_bot_path, 'r') as f:
        content = f.read()
    
    assert 'Fore.GREEN' in content and '[USER QUERY]' in content, \
        "Green colored user query logging should exist"
    print("✓ User queries logged in green (Fore.GREEN)")
    
    # Check for red final response logging
    assert 'Fore.RED' in content and '[FINAL RESPONSE]' in content, \
        "Red colored final response logging should exist"
    print("✓ Final responses logged in red (Fore.RED)")
    
    # Check for cyan eval logging
    assert 'Fore.CYAN' in content and '[EVAL]' in content, \
        "Cyan colored eval logging should exist"
    print("✓ Eval entries logged in cyan (Fore.CYAN)")
    
    # Check agent.py for colored tool calls (yellow)
    agent_path = os.path.join(os.path.dirname(__file__), '..', 'agent.py')
    with open(agent_path, 'r') as f:
        content = f.read()
    
    assert 'Fore.YELLOW' in content and '[TOOL CALL]' in content, \
        "Yellow colored tool call logging should exist"
    print("✓ Tool calls logged in yellow (Fore.YELLOW)")
    
    print("\n" + "="*60)
    print("✓ Colored logging test passed!")
    print("="*60)


def test_data_directory_setup():
    """Test that data directory is created properly."""
    print("\nTesting data directory setup...")
    print("="*60)
    
    with patch('discord_bot.discord.Client') as MockClient, \
         patch('discord_bot.Agent') as MockAgent:
        
        # Create bot instance
        bot = DiscordBot("test_token", "test_api_key")
        
        # Check that DATA_DIR and EVAL_FILE are defined
        assert hasattr(bot, 'DATA_DIR'), "Bot should have DATA_DIR attribute"
        assert hasattr(bot, 'EVAL_FILE'), "Bot should have EVAL_FILE attribute"
        
        # Check that data directory exists
        assert bot.DATA_DIR.exists(), "Data directory should be created"
        assert bot.DATA_DIR.is_dir(), "DATA_DIR should be a directory"
        
        print(f"✓ Data directory created at: {bot.DATA_DIR}")
        print(f"✓ Eval file path: {bot.EVAL_FILE}")
    
    print("\n" + "="*60)
    print("✓ Data directory setup test passed!")
    print("="*60)


def test_reaction_intents_enabled():
    """Test that reaction intents are enabled."""
    print("\nTesting reaction intents...")
    print("="*60)
    
    with patch('discord_bot.discord.Client') as MockClient, \
         patch('discord_bot.Agent') as MockAgent:
        
        # Create bot instance
        bot = DiscordBot("test_token", "test_api_key")
        
        # Check that intents.reactions was set
        # We can verify this by checking the code
        discord_bot_path = os.path.join(os.path.dirname(__file__), '..', 'discord_bot.py')
        with open(discord_bot_path, 'r') as f:
            content = f.read()
        
        assert 'intents.reactions = True' in content, \
            "Reaction intents should be enabled"
        
        print("✓ Reaction intents enabled in Discord client setup")
    
    print("\n" + "="*60)
    print("✓ Reaction intents test passed!")
    print("="*60)


async def test_log_eval_question():
    """Test the _log_eval_question method."""
    print("\nTesting eval question logging...")
    print("="*60)
    
    with patch('discord_bot.discord.Client') as MockClient, \
         patch('discord_bot.Agent') as MockAgent:
        
        # Create bot instance
        bot = DiscordBot("test_token", "test_api_key")
        
        # Create test eval file path in a temp location
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        bot.DATA_DIR = temp_dir
        bot.EVAL_FILE = temp_dir / "eval_qs.jsonl"
        
        # Create mock message
        mock_message = Mock()
        mock_message.id = 123456789
        mock_message.channel.id = 987654321
        mock_message.author.display_name = "TestUser"
        mock_message.author.id = 111111111
        mock_message.content = "What is the weather today?"
        mock_message.created_at = datetime.now()
        mock_message.add_reaction = AsyncMock()
        
        # Create mock user who added reaction
        mock_user = Mock()
        mock_user.display_name = "Tagger"
        mock_user.id = 222222222
        
        # Mock bot user
        bot.client.user = Mock()
        bot.client.user.id = 999999999
        
        # Call the method
        await bot._log_eval_question(mock_message, mock_user)
        
        # Verify the file was created and contains the entry
        assert bot.EVAL_FILE.exists(), "Eval file should be created"
        
        with open(bot.EVAL_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1, "Should have one entry"
            
            entry = json.loads(lines[0])
            assert entry['message_id'] == "123456789", "Message ID should match"
            assert entry['question'] == "What is the weather today?", "Question should match"
            assert entry['author'] == "TestUser", "Author should match"
            assert entry['tagged_by'] == "Tagger", "Tagger should match"
            assert entry['accepted_answer'] is None, "Should not have accepted answer yet"
        
        print("✓ Eval question logged correctly")
        print(f"  - Message ID: {entry['message_id']}")
        print(f"  - Question: {entry['question']}")
        print(f"  - Author: {entry['author']}")
        print(f"  - Tagged by: {entry['tagged_by']}")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    print("\n" + "="*60)
    print("✓ Eval question logging test passed!")
    print("="*60)


async def test_log_accepted_answer():
    """Test the _log_accepted_answer method."""
    print("\nTesting accepted answer logging...")
    print("="*60)
    
    with patch('discord_bot.discord.Client') as MockClient, \
         patch('discord_bot.Agent') as MockAgent:
        
        # Create bot instance
        bot = DiscordBot("test_token", "test_api_key")
        
        # Create test eval file path in a temp location
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        bot.DATA_DIR = temp_dir
        bot.EVAL_FILE = temp_dir / "eval_qs.jsonl"
        
        # Create an existing eval entry
        existing_entry = {
            "message_id": "123456789",
            "channel_id": "987654321",
            "author": "TestUser",
            "author_id": "111111111",
            "question": "What is the weather today?",
            "timestamp": datetime.now().isoformat(),
            "tagged_by": "Tagger",
            "tagged_by_id": "222222222",
            "accepted_answer": None
        }
        
        with open(bot.EVAL_FILE, 'w', encoding='utf-8') as f:
            f.write(json.dumps(existing_entry) + '\n')
        
        # Create mock bot response message (reply to original)
        mock_bot_message = Mock()
        mock_bot_message.id = 987654321
        mock_bot_message.author = Mock()
        mock_bot_message.author.id = 999999999  # Bot's ID
        mock_bot_message.content = "The weather today is sunny with a high of 75°F."
        mock_bot_message.reference = Mock()
        mock_bot_message.reference.message_id = 123456789  # Original message ID
        mock_bot_message.channel = Mock()
        mock_bot_message.add_reaction = AsyncMock()
        
        # Mock original message fetch
        mock_original_message = Mock()
        mock_original_message.id = 123456789
        mock_bot_message.channel.fetch_message = AsyncMock(return_value=mock_original_message)
        
        # Create mock user who added checkmark
        mock_user = Mock()
        mock_user.display_name = "Approver"
        mock_user.id = 333333333
        
        # Mock bot user
        bot.client.user = Mock()
        bot.client.user.id = 999999999
        
        # Call the method
        await bot._log_accepted_answer(mock_bot_message, mock_user)
        
        # Verify the entry was updated
        with open(bot.EVAL_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1, "Should still have one entry"
            
            entry = json.loads(lines[0])
            assert entry['message_id'] == "123456789", "Message ID should match"
            assert entry['accepted_answer'] == "The weather today is sunny with a high of 75°F.", \
                "Accepted answer should be set"
            assert entry['accepted_by'] == "Approver", "Approver should be set"
            assert 'accepted_at' in entry, "Accepted timestamp should be set"
        
        print("✓ Accepted answer logged correctly")
        print(f"  - Message ID: {entry['message_id']}")
        print(f"  - Question: {entry['question']}")
        print(f"  - Accepted answer: {entry['accepted_answer'][:50]}...")
        print(f"  - Accepted by: {entry['accepted_by']}")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    print("\n" + "="*60)
    print("✓ Accepted answer logging test passed!")
    print("="*60)


def test_reaction_handler_exists():
    """Test that on_reaction_add handler is registered."""
    print("\nTesting reaction handler registration...")
    print("="*60)
    
    # Check that on_reaction_add handler exists in code
    discord_bot_path = os.path.join(os.path.dirname(__file__), '..', 'discord_bot.py')
    with open(discord_bot_path, 'r') as f:
        content = f.read()
    
    assert 'async def on_reaction_add(reaction, user):' in content, \
        "on_reaction_add handler should exist"
    assert '🧪' in content, "Test tube emoji handling should exist"
    assert '✅' in content, "Check mark emoji handling should exist"
    assert '_log_eval_question' in content, "Should call _log_eval_question"
    assert '_log_accepted_answer' in content, "Should call _log_accepted_answer"
    
    print("✓ on_reaction_add handler found")
    print("✓ 🧪 (test tube) emoji handler found")
    print("✓ ✅ (check mark) emoji handler found")
    
    print("\n" + "="*60)
    print("✓ Reaction handler test passed!")
    print("="*60)


if __name__ == "__main__":
    print("Colored Logging and Reaction Tests")
    print("="*60)
    
    # Test 1: Colorama imports
    test_colorama_import()
    
    # Test 2: Colored logging implementation
    test_colored_logging_exists()
    
    # Test 3: Data directory setup
    test_data_directory_setup()
    
    # Test 4: Reaction intents
    test_reaction_intents_enabled()
    
    # Test 5: Reaction handler exists
    test_reaction_handler_exists()
    
    # Test 6: Eval question logging
    asyncio.run(test_log_eval_question())
    
    # Test 7: Accepted answer logging
    asyncio.run(test_log_accepted_answer())
    
    print("\n" + "="*60)
    print("✓ ALL COLORED LOGGING AND REACTION TESTS PASSED!")
    print("="*60)
    
    print("\n" + "="*60)
    print("Summary:")
    print("The Discord bot now has:")
    print("1. ✓ Colored logging:")
    print("   - User queries in GREEN")
    print("   - Tool calls in YELLOW")
    print("   - Final responses in RED")
    print("   - Eval entries in CYAN")
    print("2. ✓ Reaction-based evaluation logging:")
    print("   - 🧪 reaction logs user queries to data/eval_qs.jsonl")
    print("   - ✅ reaction adds accepted answers to eval entries")
    print("3. ✓ Data directory structure for eval logging")
    print("4. ✓ Proper reaction intents enabled")
    print("="*60)
