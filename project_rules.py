"""
Project Rules and Configuration System
=====================================

This module defines project-wide rules and configurations that should be followed
throughout the Portfolio Dashboard project.

Rules:
1. Always use PowerShell commands when possible
2. Handle unicode/emoji characters properly
3. Use consistent coding standards
4. Follow project structure conventions
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

class ProjectRules:
    """Project rules and configuration management."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.rules_file = self.project_root / "project_rules.json"
        self.load_rules()
    
    def load_rules(self):
        """Load project rules from configuration file."""
        default_rules = {
            "powershell_preference": True,
            "unicode_handling": {
                "enabled": True,
                "encoding": "utf-8",
                "emoji_support": True,
                "fallback_chars": "?"
            },
            "code_standards": {
                "use_black": True,
                "line_length": 88,
                "use_type_hints": True,
                "docstring_style": "google"
            },
            "project_structure": {
                "use_shared_config": True,
                "log_to_files": True,
                "cache_enabled": True
            },
            "powershell_scripts": {
                "encoding": "utf-8",
                "execution_policy": "RemoteSigned",
                "error_action": "Stop"
            }
        }
        
        if self.rules_file.exists():
            try:
                with open(self.rules_file, 'r', encoding='utf-8') as f:
                    self.rules = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load project rules: {e}")
                self.rules = default_rules
        else:
            self.rules = default_rules
            self.save_rules()
    
    def save_rules(self):
        """Save project rules to configuration file."""
        try:
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump(self.rules, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving project rules: {e}")
    
    def get_powershell_command(self, command: str) -> str:
        """Convert a command to PowerShell format if PowerShell preference is enabled."""
        if not self.rules.get("powershell_preference", True):
            return command
        
        # Convert common commands to PowerShell equivalents
        ps_commands = {
            "python": "python",
            "pip": "pip",
            "dir": "Get-ChildItem",
            "ls": "Get-ChildItem",
            "cat": "Get-Content",
            "type": "Get-Content",
            "echo": "Write-Host",
            "cd": "Set-Location",
            "pwd": "Get-Location"
        }
        
        parts = command.split()
        if parts and parts[0] in ps_commands:
            parts[0] = ps_commands[parts[0]]
            return " ".join(parts)
        
        return command
    
    def setup_unicode_environment(self):
        """Set up environment for proper unicode handling."""
        if not self.rules.get("unicode_handling", {}).get("enabled", True):
            return
        
        # Set environment variables for unicode support
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'
        os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '0'
        
        # Set console encoding for Windows
        if sys.platform == "win32":
            try:
                import locale
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            except:
                pass
    
    def run_powershell_script(self, script_content: str, args: List[str] = None) -> subprocess.CompletedProcess:
        """Run a PowerShell script with proper unicode handling."""
        if args is None:
            args = []
        
        # Create temporary script file
        script_file = self.project_root / "temp_script.ps1"
        
        try:
            # Write script with proper encoding
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Run PowerShell script
            cmd = [
                "powershell.exe",
                "-ExecutionPolicy", "RemoteSigned",
                "-File", str(script_file),
                *args
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            return result
            
        finally:
            # Clean up temporary file
            if script_file.exists():
                script_file.unlink()
    
    def create_powershell_launcher(self, script_name: str, content: str) -> Path:
        """Create a PowerShell script file with proper unicode handling."""
        script_path = self.project_root / f"{script_name}.ps1"
        
        # Add unicode handling header
        header = """# PowerShell Script with Unicode Support
# This script ensures proper unicode/emoji handling

# Set encoding for proper unicode support
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

"""
        
        full_content = header + content
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        return script_path
    
    def get_emoji_safe_string(self, text: str) -> str:
        """Ensure text is safe for unicode/emoji handling."""
        if not self.rules.get("unicode_handling", {}).get("enabled", True):
            return text
        
        try:
            # Test if the string can be encoded/decoded properly
            text.encode('utf-8').decode('utf-8')
            return text
        except UnicodeError:
            fallback_char = self.rules.get("unicode_handling", {}).get("fallback_chars", "?")
            return text.encode('utf-8', errors='replace').decode('utf-8').replace('\ufffd', fallback_char)
    
    def validate_unicode_text(self, text: str) -> Dict[str, Any]:
        """Validate unicode text and return analysis."""
        result = {
            "is_valid": True,
            "has_emojis": False,
            "encoding_issues": [],
            "safe_text": text
        }
        
        try:
            # Check for emojis
            import re
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002600-\U000026FF"  # miscellaneous symbols
                "\U00002700-\U000027BF"  # dingbats
                "]+", 
                flags=re.UNICODE
            )
            result["has_emojis"] = bool(emoji_pattern.search(text))
            
            # Test encoding
            encoded = text.encode('utf-8')
            decoded = encoded.decode('utf-8')
            result["safe_text"] = decoded
            
        except UnicodeError as e:
            result["is_valid"] = False
            result["encoding_issues"].append(str(e))
            result["safe_text"] = text.encode('utf-8', errors='replace').decode('utf-8')
        
        return result

# Global instance
project_rules = ProjectRules()

# Convenience functions
def setup_unicode():
    """Set up unicode handling for the project."""
    project_rules.setup_unicode_environment()

def get_ps_command(command: str) -> str:
    """Get PowerShell version of a command."""
    return project_rules.get_powershell_command(command)

def run_ps_script(script_content: str, args: List[str] = None) -> subprocess.CompletedProcess:
    """Run a PowerShell script."""
    return project_rules.run_powershell_script(script_content, args)

def create_ps_launcher(script_name: str, content: str) -> Path:
    """Create a PowerShell launcher script."""
    return project_rules.create_powershell_launcher(script_name, content)

def safe_unicode(text: str) -> str:
    """Get unicode-safe version of text."""
    return project_rules.get_emoji_safe_string(text)

def validate_text(text: str) -> Dict[str, Any]:
    """Validate unicode text."""
    return project_rules.validate_unicode_text(text)

if __name__ == "__main__":
    # Test the rules system
    print("🔧 Project Rules System Test")
    print("=" * 40)
    
    # Test unicode setup
    setup_unicode()
    print("✅ Unicode environment configured")
    
    # Test emoji handling
    test_text = "📊 Portfolio Dashboard 🚀 with emojis 💰"
    validation = validate_text(test_text)
    print(f"📝 Text validation: {validation}")
    
    # Test PowerShell command conversion
    ps_cmd = get_ps_command("dir *.py")
    print(f"💻 PowerShell command: {ps_cmd}")
    
    print("\n🎉 Project rules system is working!")
