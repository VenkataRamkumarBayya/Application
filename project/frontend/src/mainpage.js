import React from "react";
import { useNavigate } from "react-router-dom";

const buttonStyles = {
  base: {
    fontSize: "18px",
    padding: "25px",
    margin: "10px",
    border: "none",
    borderRadius: "12px",
    color: "#fff",
    fontWeight: 600,
    cursor: "pointer",
    width: "100%",
    height: "120px",
    textAlign: "center",
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
    transition: "transform 0.2s ease, box-shadow 0.2s ease",
  },
  ml: { backgroundColor: "#2E86DE" },
  sql: { backgroundColor: "#8e44ad" },
  react: { backgroundColor: "#00cec9" },
  oop: { backgroundColor: "#e67e22" },
  python: { backgroundColor: "#27ae60" },
  dsa: { backgroundColor: "#c0392b" },
};

const Mainpage = () => {

    const navigate = useNavigate();

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#E8ECEE" }}>
      <title>Learning Lab</title>

      {/* Top Navigation */}
      <nav
        style={{
          backgroundColor: "#3498db",
          padding: "60px 40px",
          textAlign: "center",
          color: "white",
          fontSize: "60px",
          fontWeight: "bold",
        }}
      >
        Smart Notebook
      </nav>

      {/* Grid Layout */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "30px",
          padding: "60px",
          maxWidth: "1200px",
          margin: "0 auto",
          marginTop:"100px"
        }}
      >
        <button style={{ ...buttonStyles.base, ...buttonStyles.ml }}
            onClick={() => navigate("/Mainpage2")}
        >
          🔹Machine Learning
        </button>
        <button style={{ ...buttonStyles.base, ...buttonStyles.sql }}>
          🗄️ SQL & Database Fundamentals
        </button>
        <button style={{ ...buttonStyles.base, ...buttonStyles.react }}>
          ⚛️ Web Development with React
        </button>
        <button style={{ ...buttonStyles.base, ...buttonStyles.oop }}>
          🧩 Object-Oriented Programming (OOP)
        </button>
        <button style={{ ...buttonStyles.base, ...buttonStyles.python }}>
          🐍 Python Programming Basics
        </button>
        <button style={{ ...buttonStyles.base, ...buttonStyles.dsa }}>
          📊 Data Structures & Algorithms
        </button>
      </div>
    </div>
  );
};

export default Mainpage;
