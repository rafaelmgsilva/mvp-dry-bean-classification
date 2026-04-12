const form = document.getElementById("predict-form");
const resultMessage = document.getElementById("result-message");
const warningList = document.getElementById("warning-list");
const submitButton = document.getElementById("submit-button");

async function sendPrediction(payload) {
    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });

    const data = await response.json();
    return { response, data };
}

function buildPayload(formElement) {
    const formData = new FormData(formElement);
    const payload = {};

    for (const [key, value] of formData.entries()) {
        payload[key] = Number(value);
    }

    return payload;
}

function clearWarnings() {
    warningList.innerHTML = "";
}

function renderWarnings(warnings) {
    clearWarnings();

    if (!warnings || warnings.length === 0) {
        return;
    }

    for (const warning of warnings) {
        const item = document.createElement("li");
        item.textContent = warning;
        warningList.appendChild(item);
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    submitButton.disabled = true;
    resultMessage.textContent = "Processando predição...";
    clearWarnings();

    try {
        const payload = buildPayload(form);
        const { response, data } = await sendPrediction(payload);

        if (!response.ok) {
            resultMessage.textContent = `Erro: ${data.error}`;
            return;
        }

        let message = `Classe prevista: ${data.predicted_class}`;

        if (typeof data.predicted_probability === "number") {
            const percentage = (data.predicted_probability * 100).toFixed(2);
            message += ` | Probabilidade estimada: ${percentage}%`;
        }

        resultMessage.textContent = message;
        renderWarnings(data.warnings);
    } catch (error) {
        resultMessage.textContent = "Erro de comunicação com o servidor.";
    } finally {
        submitButton.disabled = false;
    }
});