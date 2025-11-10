# --- Global variables for game state ---
current_target_color = None
i_have_won_this_round = False # Prevents spamming the "found" message

# --- Simple color-matching logic ---
def check_color_match(r, g, b, target_color):
    MIN_CLEAR = 60 
    c = r + g + b
    if c < MIN_CLEAR:
        return False

    RATIO = 1.45

    if target_color == "RED":
        return (r > g * RATIO) and (r > b * RATIO)
    if target_color == "GREEN":
        return (g > r * RATIO) and (g > b * RATIO)
    if target_color == "BLUE":
        return (b > r * RATIO) and (b > g * RATIO)
    return False

# --- Callback for messages from the Game Master ---
def on_message(client, userdata, msg):
    global current_target_color, i_have_won_this_round
    
    payload = msg.payload.decode()
    
    if msg.topic == MASTER_TOPIC:
        # A new round has started!
        current_target_color = payload
        i_have_won_this_round = False # Reset our "won" flag
        print(f"*** New Target: Find [{current_target_color}]! ***")
        
    elif msg.topic == WINNER_TOPIC:
        # Round is over
        print(f"*** Game Status: {payload} ***")
        current_target_color = None # Stop sensing until next round

# --- Main Sensing Loop ---
try:
    while True:
        # Only check the sensor if there is a target and we haven't won yet
        if current_target_color is not None and not i_have_won_this_round:
            
            r, g, b, c = apds.color_data
            
            if check_color_match(r, g, b, current_target_color):
                print(f"MATCH FOUND! I see {current_target_color}!")
                
                # Set our flag to true so we don't send multiple messages
                i_have_won_this_round = True 
                
                # Publish our "I WON" message!
                client.publish(PLAYER_TOPIC, PLAYER_NAME)
        
        time.sleep(0.1) # Loop delay