import { useEffect, useState } from "react";
import axios from "axios";

export default function App() {
  const [state, setState] = useState("UNKNOWN");
  const [ultra, setUltra] = useState<number | null>(null);

  // Poll backend for states
  useEffect(() => {
    const t = setInterval(async () => {
      try {
        const s = await axios.get("/api/state");
        setState(s.data.state);

        const u = await axios.get("/api/ultrasonic");
        setUltra(u.data.distance);
      } catch (e) {
        console.log(e);
      }
    }, 400);
    return () => clearInterval(t);
  }, []);

  const startPick = async () => {
    await axios.post("/api/start");
  };

  const stopRobot = async () => {
    await axios.post("/api/stop");
  };

  return (
    <div style={{ padding: 40, fontFamily: "sans-serif" }}>
      <h1>🤖 VLA Orchestrator Dashboard</h1>

      <h2>Robot State: <span style={{ color: "green" }}>{state}</span></h2>
      <h3>Ultrasonic Distance: 
        <span style={{ color: "blue" }}>
          {ultra !== null ? ultra.toFixed(3) + " m" : "N/A"}
        </span>
      </h3>

      <button onClick={startPick} style={{ padding: 20, fontSize: 18 }}>
        ▶️ Start Pick & Place
      </button>

      <button onClick={stopRobot} style={{
        padding: 20,
        marginLeft: 20,
        backgroundColor: "red",
        color: "white",
        fontSize: 18
      }}>
        🛑 EMERGENCY STOP
      </button>
    </div>
  );
}
