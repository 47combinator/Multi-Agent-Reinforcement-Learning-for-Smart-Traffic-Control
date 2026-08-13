import os
import sys
import eventlet
eventlet.monkey_patch() # Must be called very early

from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from pathlib import Path

# Setup Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.runners import run_agent_stream

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

state = {
    "simulation_running": False,
    "should_stop": False,
    "is_paused": False
}

@app.route("/api/status")
def status():
    return jsonify({
        "status": "ok", 
        "running": state["simulation_running"],
        "paused": state["is_paused"]
    })

@app.route("/api/stop", methods=["POST"])
def stop_simulation():
    if state["simulation_running"]:
        state["should_stop"] = True
        return jsonify({"status": "stopping"})
    return jsonify({"status": "not_running"})

@app.route("/api/pause", methods=["POST"])
def pause_simulation():
    if state["simulation_running"]:
        state["is_paused"] = True
        return jsonify({"status": "paused"})
    return jsonify({"status": "not_running", "error": "Cannot pause when not running."})

@app.route("/api/resume", methods=["POST"])
def resume_simulation():
    if state["simulation_running"] and state["is_paused"]:
        state["is_paused"] = False
        return jsonify({"status": "resumed"})
    return jsonify({"status": "not_paused", "error": "Simulation is not paused."})

@socketio.on("connect")
def on_connect():
    print("Client connected")

@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected")

@socketio.on("start_simulation")
def handle_start_simulation(data):
    if state["simulation_running"]:
        socketio.emit("error", {"message": "Simulation already running"})
        return
    
    state["simulation_running"] = True
    state["should_stop"] = False
    
    model_type = data.get("model", "PPO")
    episodes = int(data.get("episodes", 1))
    seed = int(data.get("seed", 42))
    use_gui = data.get("use_gui", False)
    
    print(f"Starting {model_type} simulation for {episodes} episodes...")
    
    try:
        # Run agent stream is a generator that yields metrics
        for metrics in run_agent_stream(model_type, episodes, seed, use_gui, state):
            # Pause handling
            while state.get("is_paused", False):
                socketio.sleep(0.1)
                if state.get("should_stop", False):
                    break
            
            if state.get("should_stop", False):
                break
                
            socketio.emit("metrics", metrics)
            socketio.sleep(0.01) # Yield to eventlet
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in simulation: {e}")
        socketio.emit("error", {"message": str(e)})
    finally:
        state["simulation_running"] = False
        socketio.emit("simulation_complete", {"message": "Done"})
        print("Simulation complete.")

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
