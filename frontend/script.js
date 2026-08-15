// 🔍 Analyze button
document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("audioFile");
  const resultDiv = document.getElementById("result");

  if (!fileInput.files.length) {
    resultDiv.innerHTML = "⚠️ Please select an audio file.";
    return;
  }

  const file = fileInput.files[0];

  if (!file.type.startsWith("audio/")) {
    resultDiv.innerHTML = "⚠️ Please select a valid audio file.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  resultDiv.innerHTML = "⏳ Analyzing audio... Please wait...";

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (data.result) {
      resultDiv.innerHTML = `🧠 <b>Prediction Result:</b> ${data.result}`;
    } else {
      resultDiv.innerHTML = `❌ ${data.error}`;
    }

  } catch (error) {
    resultDiv.innerHTML = "❌ Server error!";
  }
});

// 🎵 Audio Preview (NEW)
document.getElementById("audioFile").addEventListener("change", function () {
  const previewDiv = document.getElementById("audioPreview");
  const resultDiv = document.getElementById("result");

  previewDiv.innerHTML = "";

  if (this.files.length > 0) {
    const file = this.files[0];

    // Show name
    resultDiv.innerHTML = `🎵 Selected file: <b>${file.name}</b>`;

    // Create audio player
    const audioURL = URL.createObjectURL(file);

    previewDiv.innerHTML = `
      <audio controls style="margin-top:10px; width:100%; border-radius:10px;">
        <source src="${audioURL}" type="${file.type}">
        Your browser does not support the audio tag.
      </audio>
    `;
  }
});
