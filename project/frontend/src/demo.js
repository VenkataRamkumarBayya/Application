import React, { useEffect, useRef, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";


import { useNavigate } from "react-router-dom";
import { pages } from "./pages";

const SmartNotebook = () => {
  const [pageIndex, setPageIndex] = useState(0);
  const containerRef = useRef(null);
  const leftPanelRef = useRef(null);
  const resizerRef = useRef(null);
  const [isResizing, setIsResizing] = useState(false);

const navigate=useNavigate();

  useEffect(() => {
    if (containerRef.current) {
      addCell();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing || !leftPanelRef.current) return;
      const newWidth = Math.max(300, e.clientX);
      leftPanelRef.current.style.width = `${newWidth}px`;
    };

    const handleMouseUp = () => setIsResizing(false);

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing]);

  const addCell = () => {
    const container = containerRef.current;
    if (!container) return;

    const cell = document.createElement("div");
    cell.className = "cell mb-4 p-3 rounded";
    cell.style.background = "white";
    cell.style.border = "1px solid #333";

    const row = document.createElement("div");
    row.className = "d-flex align-items-start";

    const btnCol = document.createElement("div");
    btnCol.className = "d-flex flex-column gap-2 me-3 pt-1";

    const runBtn = document.createElement("button");
    runBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
    runBtn.style.fontSize = "24px";
    runBtn.style.background = "none";
    runBtn.style.border = "none";
    runBtn.style.color = "black"; 
    runBtn.style.cursor = "pointer";
    runBtn.className = "btn text-gray p-0";

    const deleteBtn = document.createElement("button");
    deleteBtn.innerHTML = '<i class="fa-solid fa-trash"></i>';
    deleteBtn.className = "btn text-gray p-0";
    deleteBtn.style.fontSize = "22px";
    deleteBtn.style.color = "black";
    deleteBtn.style.cursor = "pointer";
    deleteBtn.style.background = "none";
    deleteBtn.onclick = () => container.removeChild(cell);

    btnCol.appendChild(runBtn);
    btnCol.appendChild(deleteBtn);

    const textarea = document.createElement("textarea");
    textarea.placeholder = "# Write your Python code here...";
    textarea.className = "form-control text- bg-light";
    textarea.style.height = "120px";
    textarea.style.border = "2px solid #ccc";
    textarea.style.fontFamily = "'Fira Code', monospace";
    textarea.style.fontSize = "16px";

    const textCol = document.createElement("div");
    textCol.className = "flex-grow-1";
    textCol.appendChild(textarea);

    row.appendChild(btnCol);
    row.appendChild(textCol);

    const output = document.createElement("pre");
    output.textContent = "Output will appear here.";
    output.className = "p-3 mt-3 rounded text-white";
    output.style.background = "#0d1117";
    output.style.fontFamily = "'Fira Code', monospace";
    output.style.fontSize = "15px";

    const addBelowBtn = document.createElement("button");
    addBelowBtn.textContent = "➕ Add cell";
    addBelowBtn.className = "btn btn-outline-dark btn-sm mt-3";
    addBelowBtn.onclick = () => addCell();

    runBtn.onclick = () => runCode(textarea.value, output);

    cell.appendChild(row);
    cell.appendChild(output);
    cell.appendChild(addBelowBtn);

    container.appendChild(cell);
  };

  const runCode = (code, output) => {
    output.textContent = "Running...";
    fetch("http://127.0.0.1:8000/run-code", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    })
      .then((res) => res.json())
      .then((data) => {
        const out = (data.output || "") + (data.error || "");
        output.textContent = out.trim() || "(No output)";
      })
      .catch((err) => {
        output.textContent = "Error: " + err.message;
      });
  };

  return (
    <div className="layout d-flex" style={{ height: "100vh", overflow: "hidden", background: "#282c34", color: "white" }}>
      {/* Left Panel */}
      <div
        ref={leftPanelRef}
        className="left p-4"
        style={{
          width: "700px",
          background: "#282c34",
          overflowY: "auto",  
          borderRight: "1px solid #333",
        }}
      >
        <nav className="d-flex align-items-center justify-content-between mb-4">
          <div style={{ marginBottom: "20px", marginTop:"25px" }}>
          <button
            onClick={() => navigate("/mainpage2")}
            style={{
              backgroundColor: "#007bff",
              color: "#fff",
              border: "none",
              padding: "8px 16px",
              borderRadius: "8px",
              fontSize: "1rem",
              cursor: "pointer",
              boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <i className="fa-solid fa-arrow-left"></i> 
          </button>
        </div>
          <h2 className="mb-0"> Building ML Models</h2>
          <div className="d-flex gap-2">
            <button
              className="btn btn-outline-light btn-sm"
              onClick={() => setPageIndex(Math.max(0, pageIndex - 1))}
              disabled={pageIndex === 0}
            >
              <i className="fa-solid fa-angle-left"></i>
            </button>
            <button
              className="btn btn-outline-light btn-sm"
              onClick={() => setPageIndex(Math.min(pages.length - 1, pageIndex + 1))}
              disabled={pageIndex === pages.length - 1}
            >
              <i className="fa-solid fa-angle-right"></i>
            </button>
          </div>
        </nav>

        <h5 className="text-info">{pages[pageIndex].title}</h5>
        <div>{pages[pageIndex].content}</div>
      </div>

      {/* Resizer Divider */}
      <div
        ref={resizerRef}
        onMouseDown={() => setIsResizing(true)}
        style={{
          width: "5px",
          cursor: "col-resize",
          background: "#444",
          zIndex: 1,
        }}
      ></div>

      {/* Right Panel */}
      <div className="right p-4" style={{ flex: 1, overflowY: "auto" }}>
        <div className="d-flex justify-content-between align-items-center mb-4">
          <div>
            <button
              className="btn btn-outline-success me-2"
              onClick={() => {
                const cells = containerRef.current?.querySelectorAll(".cell");
                cells?.forEach((cell) => {
                  const textarea = cell.querySelector("textarea");
                  const output = cell.querySelector("pre");
                  if (textarea && output) runCode(textarea.value, output);
                });
              }}
            >
              <i className="fa-solid fa-play me-2"></i>All
            </button>

            <button className="btn btn-outline-light me-2" onClick={addCell}>
              ➕ Code
            </button>

            <button
              className="btn btn-outline-danger"
              onClick={() => {
                containerRef.current.innerHTML = "";
              }}
            >
              <i className="fa-solid fa-trash me-1"></i>All
            </button>
          </div>
        </div>

        <div ref={containerRef}></div>
      </div>
    </div>
  );
};

export default SmartNotebook;
