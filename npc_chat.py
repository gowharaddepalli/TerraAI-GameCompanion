import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

# Try to import OpenAI
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_AVAILABLE = True
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False
    client = None

HELP_KEYWORDS = {"help", "where", "how", "quest", "please", "assist", "lost"}
INSULT_KEYWORDS = {"stupid", "idiot", "useless", "dumb", "hate", "shut up"}

class PlayerState:
    def __init__(self):
        self.mood = "neutral"
        self.last_messages = []

def parse_timestamp(ts: str) -> datetime:
    """Parse ISO 8601 timestamp"""
    ts = ts.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(ts)

def update_mood(text: str, current_mood: str) -> Tuple[str, str]:
    """Update NPC mood based on player message"""
    text_lower = text.lower()
    
    # Check for insults
    if any(insult in text_lower for insult in INSULT_KEYWORDS):
        return "angry", "insult_detected"
    
    # Check for help requests
    if any(help_word in text_lower for help_word in HELP_KEYWORDS) or text_lower.endswith("?"):
        return "friendly", "help_or_question"
    
    # De-escalate from angry
    if current_mood == "angry" and any(word in text_lower for word in ["sorry", "please", "thanks"]):
        return "neutral", "deescalated"
    
    return current_mood, "no_change"

def build_messages(npc_name: str, mood: str, history: List[str], current_text: str) -> List[dict]:
    """Build messages for OpenAI API"""
    system_prompt = f"""You are {npc_name}, a game NPC. Current mood: {mood}.
    Keep replies short (1-2 sentences). Be helpful if friendly, concise if neutral, curt if angry."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    if history:
        context = "Previous messages:\n" + "\n".join(f"- {msg}" for msg in history)
        messages.append({"role": "user", "content": context})
    
    # Add current message
    messages.append({"role": "user", "content": current_text})
    
    return messages

def rule_based_reply(mood: str, text: str) -> str:
    """Fallback reply generator"""
    text_lower = text.lower()
    
    if mood == "angry":
        if "sorry" in text_lower:
            return "Fine. Head east to the mill."
        return "Make it quick. East road leads to the mill."
    
    if mood == "friendly":
        if any(word in text_lower for word in ["where", "go", "quest"]):
            return "Follow the east road to the mill. The elder can help you!"
        return "How can I help you today?"
    
    # Neutral mood
    if any(word in text_lower for word in ["where", "go", "quest"]):
        return "East road goes to the mill. Village center is by the well."
    return "What do you need? I can offer directions or information."

def generate_reply(npc_name: str, mood: str, history: List[str], text: str, model: str) -> Tuple[str, str]:
    """Generate NPC reply using OpenAI or fallback"""
    if OPENAI_AVAILABLE and client:
        try:
            messages = build_messages(npc_name, mood, history, text)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=60,
                temperature=0.7
            )
            reply = response.choices[0].message.content.strip()
            return reply, f"openai:{model}"
        except Exception:
            pass  # Fall through to rule-based
    
    return rule_based_reply(mood, text), "fallback"

def process_messages(input_file: str, log_file: str, model: str, npc_name: str):
    """Process all player messages"""
    # Load and sort messages
    with open(input_file, 'r') as f:
        messages = json.load(f)
    
    messages.sort(key=lambda m: parse_timestamp(m["timestamp"]))
    
    # Initialize player states
    player_states: Dict[int, PlayerState] = {}
    
    # Prepare log file
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_handle = open(log_file, 'w')
    else:
        log_handle = None
    
    # Process each message
    for msg in messages:
        player_id = msg["player_id"]
        text = msg["text"]
        timestamp = msg["timestamp"]
        
        # Get or create player state
        if player_id not in player_states:
            player_states[player_id] = PlayerState()
        
        state = player_states[player_id]
        context = state.last_messages[-3:]  # Last 3 messages
        
        # Update mood
        new_mood, reason = update_mood(text, state.mood)
        state.mood = new_mood
        
        # Generate reply
        reply, source = generate_reply(npc_name, state.mood, context, text, model)
        
        # Update message history
        state.last_messages.append(text)
        if len(state.last_messages) > 3:
            state.last_messages = state.last_messages[-3:]
        
        # Create log entry
        log_entry = {
            "player_id": player_id,
            "timestamp": timestamp,
            "message_text": text,
            "npc_reply": reply,
            "state_used": context,
            "npc_mood": state.mood,
            "mood_reason": reason,
            "model_source": source,
            "npc_name": npc_name
        }
        
        # Output to console
        print(json.dumps(log_entry))
        
        # Write to log file
        if log_handle:
            log_handle.write(json.dumps(log_entry) + '\n')
    
    if log_handle:
        log_handle.close()

def main():
    parser = argparse.ArgumentParser(description="NPC Chat System")
    parser.add_argument("input_file", help="Input JSON file with player messages")
    parser.add_argument("--log", default="logs/chat_log.jsonl", help="Output log file")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--npc-name", default="Elya the Ranger", help="NPC name")
    
    args = parser.parse_args()
    
    if not OPENAI_AVAILABLE:
        print("OpenAI not available - using rule-based fallback mode")
    
    process_messages(args.input_file, args.log, args.model, args.npc_name)

if __name__ == "__main__":
    main()