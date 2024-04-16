async function saveFile() {
    let formData = new FormData()
    formData.append("file", fileupload.files[0]);
    let resp = await fetch("http://localhost:8000/predict", {method: "POST", body: formData});
    resp = await resp.json();
    let prediction = document.getElementById("prediction");
    let confidence = document.getElementById("confidence");

    prediction.innerHTML = resp.class;
    confidence.innerHTML = "Confidence: " +  resp.confidence;

    console.log("Class: " + resp.class + " " + "Confidence: " + resp.confidence);
}

function preview(event) {
    if (event.target.files.length > 0) {
        var src = URL.createObjectURL(event.target.files[0]);
        var preview = document.getElementById("file-preview");
        preview.src = src;
        preview.style.display = "block";
    }
}
