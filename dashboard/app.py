"""
🎓 CA2 GROUP ASSIGNMENT LAUNCHER
Run this script to launch both dashboards simultaneously.
"""

import subprocess
import sys
import time
import webbrowser
import os

# --- CONFIGURATION ---
# Make sure these match the filenames exactly
file_thomas = "ThomasDashboard.py"
file_lingger = "LinggerDashboard.py"

# Make sure these match the ports inside those files
port_thomas = 8050
port_lingger = 8051

def run_dashboard(filename):
    """Starts a dashboard in a separate process"""
    print(f"   ⏳ Starting {filename}...")
    # sys.executable ensures we use the same Python environment currently running
    return subprocess.Popen([sys.executable, filename])

if __name__ == "__main__":
    print("\n" + "="*60)
    print("      🚀 INITIALIZING EDU-ANALYTICS DASHBOARD SUITE")
    print("="*60 + "\n")

    # 1. Launch Processes
    process_thomas = run_dashboard(file_thomas)
    process_lingger = run_dashboard(file_lingger)

    # 2. Wait a moment for servers to spin up
    print("\n   ... waiting for servers to initialize (5 seconds) ...\n")
    time.sleep(5)

    # 3. Print The Control Panel
    print("="*60)
    print("✅  ALL SYSTEMS ONLINE")
    print("="*60)
    
    link_thomas = f"http://127.0.0.1:{port_thomas}/"
    link_lingger = f"http://127.0.0.1:{port_lingger}/"

    print(f"\n🔹 [Student 1] Thomas (Manager View):")
    print(f"    👉 {link_thomas}")

    print(f"\n🔹 [Student 2] Lingger (Counselor View):")
    print(f"    👉 {link_lingger}")
    
    print("\n" + "="*60)
    print("PRESS 'CTRL + C' IN THIS TERMINAL TO STOP BOTH DASHBOARDS")
    print("="*60)

    # 4. Optional: Auto-open in browser (Uncomment if you want this)
    # webbrowser.open(link_thomas)
    # webbrowser.open(link_lingger)

    try:
        # Keep script running to maintain subprocesses
        process_thomas.wait()
        process_lingger.wait()
    except KeyboardInterrupt:
        print("\n\n🛑  SHUTTING DOWN ALL DASHBOARDS...")
        process_thomas.terminate()
        process_lingger.terminate()
        print("✅  Shutdown Complete. Goodbye!")