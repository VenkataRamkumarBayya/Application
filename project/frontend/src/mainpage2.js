import React from "react";
import { useNavigate, Link } from "react-router-dom";

const topics = [
  {
    title: "Building ML Models",
    description: "Construct, train, evaluate, and improve machine learning models end to end.",
    link: "/notebook",
  },
  {
    title: "Introduction to Neural Networks",
    description: "Understand and build basic neural networks for classification tasks.",
  },
  {
    title: "Convolutional Neural Networks",
    description: "Explore CNNs for image classification and computer vision.",
  },
  {
    title: "Recurrent Neural Networks",
    description: "Model sequential data like text and time series with RNNs and LSTMs.",
  },
  {
    title: "Time Series Forecasting",
    description: "Learn methods to predict time-dependent patterns and trends.",
  },
  {
    title: "Transfer Learning",
    description: "Leverage pretrained models to quickly build powerful solutions.",
  },
  {
    title: "Model Monitoring and Drift Detection",
    description: "Track production performance and detect when retraining is needed.",
  },
];

const Mainpage2 = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = React.useState("");
  const [currentPage, setCurrentPage] = React.useState(1);
  const topicsPerPage = 6;

  const filteredTopics = topics.filter((topic) =>
    topic.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const totalPages = Math.ceil(filteredTopics.length / topicsPerPage);

  const currentTopics = filteredTopics.slice(
    (currentPage - 1) * topicsPerPage,
    currentPage * topicsPerPage
  );

  const goToPage = (pageNumber) => {
    if (pageNumber >= 1 && pageNumber <= totalPages) {
      setCurrentPage(pageNumber);
    }
  };

  return (
    <div
      style={{
        background: "linear-gradient(to right, #dbeafe, #fefce8)", // blue to yellow gradient
        minHeight: "100vh",
        padding: "40px 0",
      }}
    >
      <div
        className="container py-5"
        style={{
          backgroundColor: "#ffffff",
          borderRadius: "16px",
          boxShadow: "0 0 20px rgba(0,0,0,0.1)",
          padding: "30px",
        }}
      >
        {/* ✅ Back Button */}
        <div style={{ marginBottom: "20px" }}>
          <button
            onClick={() => navigate("/")}
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

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "30px",
          }}
        >
          <h1
            className="mb-0"
            style={{ fontWeight: "600", color: "#343a40", fontFamily: "sans-serif" }}
          >
            ML Lab
          </h1>

          <div
            style={{
              position: "relative",
              maxWidth: "400px",
              marginLeft: "auto",
              width: "100%",
            }}
          >
            <input
              type="search"
              className="form-control"
              placeholder="Search topics by title..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              style={{
                width: "100%",
                padding: "10px 15px 10px 40px", // space for icon
                fontSize: "1rem",
                borderRadius: "12px",
                border: "1px solid #ccc",
                boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
              }}
            />
            <span
              style={{
                position: "absolute",
                left: "12px",
                top: "50%",
                transform: "translateY(-50%)",
                color: "#888",
                fontSize: "18px",
                pointerEvents: "none",
              }}
            >
              <i className="fa-solid fa-search"></i>
            </span>
          </div>
        </div>

        <p className="text-muted mb-4 fs-5">
          Explore hands-on modules covering machine learning and deep learning concepts.
        </p>

        <div className="row g-4">
          {currentTopics.length > 0 ? (
            currentTopics.map((topic, index) => (
              <div key={index} className="col-md-6 col-lg-4">
                <div
                  className="card h-100 border-0 shadow-sm"
                  style={{ borderRadius: "16px", backgroundColor: "#ffffff" }}
                >
                  <div className="card-body d-flex flex-column">
                    <h5 className="card-title" style={{ color: "#2c3e50" }}>
                      {topic.title}
                    </h5>
                    <p className="card-text flex-grow-1" style={{ color: "#555" }}>
                      {topic.description}
                    </p>
                    {topic.link ? (
                      <Link
                        to={topic.link}
                        className="btn btn-primary mt-3"
                        style={{ borderRadius: "8px" }}
                      >
                        Start Learning
                      </Link>
                    ) : (
                      <button
                        className="btn btn-secondary mt-3"
                        disabled
                        title="Coming soon"
                        style={{ borderRadius: "8px" }}
                      >
                        Coming Soon
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="text-muted">No topics found.</div>
          )}
        </div>

        {/* Pagination Controls */}
        {totalPages > 1 && (
          <div
            className="d-flex justify-content-center align-items-center gap-2"
            style={{
              marginTop: "100px",
              flexWrap: "wrap",
            }}
          >
            <button
              className="btn btn-primary px-4"
              onClick={() => goToPage(currentPage - 1)}
              disabled={currentPage === 1}
              style={{ borderRadius: "25px" }}
            >
              Prev
            </button>

            {Array.from({ length: totalPages }, (_, i) => (
              <button
                key={i}
                className={`btn ${
                  currentPage === i + 1 ? "btn-primary" : "btn-outline-primary"
                } px-3`}
                onClick={() => goToPage(i + 1)}
                style={{
                  borderRadius: "50%",
                  minWidth: "40px",
                  height: "40px",
                  padding: 0,
                }}
              >
                {i + 1}
              </button>
            ))}

            <button
              className="btn btn-primary px-4"
              onClick={() => goToPage(currentPage + 1)}
              disabled={currentPage === totalPages}
              style={{ borderRadius: "25px" }}
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Mainpage2;
