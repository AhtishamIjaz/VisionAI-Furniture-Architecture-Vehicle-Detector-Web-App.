document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzoneContent = document.getElementById("dropzoneContent");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const removeImgBtn = document.getElementById("removeImgBtn");
  const sampleGrid = document.getElementById("sampleGrid");

  const modelSelect = document.getElementById("modelSelect");
  const confSlider = document.getElementById("confSlider");
  const confVal = document.getElementById("confVal");
  const predictBtn = document.getElementById("predictBtn");

  const statusMessage = document.getElementById("statusMessage");
  const viewToggles = document.getElementById("viewToggles");
  const loadingSpinner = document.getElementById("loadingSpinner");
  const emptyPlaceholder = document.getElementById("emptyPlaceholder");
  const singleView = document.getElementById("singleView");
  const splitView = document.getElementById("splitView");
  const resultImage = document.getElementById("resultImage");
  const splitOriginal = document.getElementById("splitOriginal");
  const splitAnnotated = document.getElementById("splitAnnotated");

  const hudStats = document.getElementById("hudStats");
  const statTotal = document.getElementById("statTotal");
  const statFurniture = document.getElementById("statFurniture");
  const statArchitecture = document.getElementById("statArchitecture");
  const statVehicles = document.getElementById("statVehicles");
  const statTime = document.getElementById("statTime");

  const detectionsSection = document.getElementById("detectionsSection");
  const detectionsTableBody = document.getElementById("detectionsTableBody");
  const downloadImgBtn = document.getElementById("downloadImgBtn");
  const exportJsonBtn = document.getElementById("exportJsonBtn");

  // State
  let currentFile = null;
  let currentSampleName = null;
  let lastPredictionData = null;
  let activeViewMode = "annotated";

  // 1. Confidence Slider Listener
  confSlider.addEventListener("input", (e) => {
    confVal.textContent = `${e.target.value}%`;
  });

  // 2. Dropzone & File Selection
  dropzone.addEventListener("click", (e) => {
    if (e.target !== removeImgBtn) {
      fileInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  removeImgBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetImageSelection();
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file (JPG, PNG, WEBP).");
      return;
    }
    currentFile = file;
    currentSampleName = null;

    // Deselect sample cards
    document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropzoneContent.style.display = "none";
      previewContainer.style.display = "flex";
      predictBtn.disabled = false;
      predictBtn.innerHTML = '<span class="btn-icon">⚡</span> Run Prediction on Picture';
    };
    reader.readAsDataURL(file);
  }

  function resetImageSelection() {
    currentFile = null;
    currentSampleName = null;
    fileInput.value = "";
    imagePreview.src = "";
    dropzoneContent.style.display = "block";
    previewContainer.style.display = "none";
    predictBtn.disabled = true;
    document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
  }

  // 3. Load Samples
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      sampleGrid.innerHTML = "";

      if (data.samples && data.samples.length > 0) {
        data.samples.forEach(sample => {
          const card = document.createElement("div");
          card.className = "sample-card";
          card.innerHTML = `
            <img src="${sample.url}" alt="${sample.title}" class="sample-thumb">
            <div>
              <div class="sample-title">${sample.title}</div>
              <div class="sample-desc">Click to load</div>
            </div>
          `;
          card.addEventListener("click", () => {
            selectSample(sample, card);
          });
          sampleGrid.appendChild(card);
        });
      } else {
        sampleGrid.innerHTML = '<p style="grid-column: span 2; font-size: 0.76rem; color: #64748b;">Upload your own downloaded pictures above.</p>';
      }
    } catch (err) {
      console.warn("Could not fetch samples:", err);
    }
  }

  function selectSample(sample, cardElement) {
    currentFile = null;
    currentSampleName = sample.filename;
    fileInput.value = "";

    document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
    cardElement.classList.add("active");

    imagePreview.src = sample.url;
    dropzoneContent.style.display = "none";
    previewContainer.style.display = "flex";
    predictBtn.disabled = false;
    predictBtn.innerHTML = `<span class="btn-icon">⚡</span> Predict: ${sample.title}`;
  }

  // 4. Run Prediction Request
  predictBtn.addEventListener("click", runPrediction);

  async function runPrediction() {
    if (!currentFile && !currentSampleName) {
      alert("Please choose or upload a downloaded picture first.");
      return;
    }

    // UI Loading state
    loadingSpinner.style.display = "flex";
    predictBtn.disabled = true;
    statusMessage.textContent = "AI model is analyzing items in the image...";

    const formData = new FormData();
    const confValDecimal = parseFloat(confSlider.value) / 100.0;
    formData.append("conf", confValDecimal);
    formData.append("model", modelSelect.value);

    if (currentFile) {
      formData.append("file", currentFile);
    } else if (currentSampleName) {
      formData.append("sample", currentSampleName);
    }

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Failed to predict image.");
      }

      lastPredictionData = data;
      renderResults(data);

    } catch (err) {
      console.error(err);
      alert("Prediction error: " + err.message);
      statusMessage.textContent = "Error during prediction. Please try again.";
    } finally {
      loadingSpinner.style.display = "none";
      predictBtn.disabled = false;
    }
  }

  // 5. Render Prediction Results
  function renderResults(data) {
    emptyPlaceholder.style.display = "none";
    viewToggles.style.display = "flex";
    hudStats.style.display = "grid";
    detectionsSection.style.display = "block";

    statusMessage.textContent = `Completed in ${data.inference_time_ms} ms with ${data.model_used}`;

    // HUD Stats
    statTotal.textContent = data.total_objects;
    statFurniture.textContent = (data.category_counts && data.category_counts["Furniture"]) || 0;
    statArchitecture.textContent = (data.category_counts && data.category_counts["Architecture / Interior"]) || 0;
    statVehicles.textContent = (data.category_counts && data.category_counts["Vehicles"]) || 0;
    statTime.textContent = `${data.inference_time_ms} ms`;

    // Images
    resultImage.src = data.annotated_image;
    splitOriginal.src = data.original_image;
    splitAnnotated.src = data.annotated_image;

    updateViewDisplay();

    // Predictions Table
    detectionsTableBody.innerHTML = "";
    if (data.detections && data.detections.length > 0) {
      data.detections.forEach((item, idx) => {
        const tr = document.createElement("tr");

        let catClass = "cat-other";
        if (item.category === "Furniture") catClass = "cat-furniture";
        else if (item.category === "Architecture / Interior") catClass = "cat-architecture";
        else if (item.category === "Vehicles") catClass = "cat-vehicles";

        tr.innerHTML = `
          <td><strong>#${idx + 1}</strong></td>
          <td><span class="cat-pill ${catClass}">${item.category}</span></td>
          <td><strong style="color: #f8fafc;">${item.label}</strong></td>
          <td>
            <div class="conf-meter">
              <span style="min-width: 35px; font-weight: 600;">${item.confidence_pct}%</span>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width: ${item.confidence_pct}%"></div>
              </div>
            </div>
          </td>
          <td style="font-family: monospace; font-size: 0.72rem; color: #94a3b8;">
            [${item.bbox[0]}, ${item.bbox[1]}, ${item.bbox[2]}, ${item.bbox[3]}]
          </td>
        `;
        detectionsTableBody.appendChild(tr);
      });
    } else {
      detectionsTableBody.innerHTML = `
        <tr>
          <td colspan="5" style="text-align: center; color: #94a3b8; padding: 24px;">
            No objects detected above ${confSlider.value}% confidence. Try lowering the threshold slider.
          </td>
        </tr>
      `;
    }
  }

  // 6. View Mode Toggles
  document.querySelectorAll(".btn-toggle").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".btn-toggle").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeViewMode = btn.dataset.view;
      updateViewDisplay();
    });
  });

  function updateViewDisplay() {
    if (!lastPredictionData) return;

    if (activeViewMode === "split") {
      singleView.style.display = "none";
      splitView.style.display = "grid";
    } else {
      splitView.style.display = "none";
      singleView.style.display = "block";
      if (activeViewMode === "original") {
        resultImage.src = lastPredictionData.original_image;
      } else {
        resultImage.src = lastPredictionData.annotated_image;
      }
    }
  }

  // 7. Download & Export
  downloadImgBtn.addEventListener("click", () => {
    if (!lastPredictionData || !lastPredictionData.annotated_image) return;
    const a = document.createElement("a");
    a.href = lastPredictionData.annotated_image;
    a.download = `predicted_objects_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });

  exportJsonBtn.addEventListener("click", () => {
    if (!lastPredictionData) return;
    const jsonReport = {
      timestamp: new Date().toISOString(),
      model: lastPredictionData.model_used,
      total_objects: lastPredictionData.total_objects,
      category_counts: lastPredictionData.category_counts,
      inference_time_ms: lastPredictionData.inference_time_ms,
      detections: lastPredictionData.detections
    };
    const blob = new Blob([JSON.stringify(jsonReport, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `detection_report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // Initialize
  loadSamples();
});
