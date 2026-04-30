let currentJobId = null;

const jobOutput = document.querySelector('#job-output');
const runButton = document.querySelector('#run-job');
const askButton = document.querySelector('#ask-button');
const answerOutput = document.querySelector('#answer-output');

async function fetchJson(url, options) {
  let payload = null;
  let response;

  try {
    response = await fetch(url, options);
    payload = await response.json();
  } catch (error) {
    throw new Error(`Request failed: ${error.message}`);
  }

  if (!response.ok) {
    throw new Error(payload?.detail || payload?.error || `Request failed with status ${response.status}`);
  }

  return payload;
}

document.querySelector('#job-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const body = {
    input_path: document.querySelector('#input-path').value,
    source_language: document.querySelector('#source-language').value,
    translation_target: document.querySelector('#translation-target').value || null,
    subtitles: document.querySelector('#subtitles').checked,
    face_blur: document.querySelector('#face-blur').checked
  };

  try {
    const payload = await fetchJson('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!payload.id) {
      throw new Error('Job response did not include an id.');
    }

    currentJobId = payload.id;
    jobOutput.textContent = JSON.stringify(payload, null, 2);
    runButton.disabled = false;
    askButton.disabled = true;
  } catch (error) {
    currentJobId = null;
    runButton.disabled = true;
    askButton.disabled = true;
    jobOutput.textContent = error.message;
  }
});

runButton.addEventListener('click', async () => {
  runButton.disabled = true;
  askButton.disabled = true;
  jobOutput.textContent = 'Processing...';

  try {
    const payload = await fetchJson(`/api/jobs/${currentJobId}/run`, { method: 'POST' });
    if (payload.status === 'failed') {
      jobOutput.textContent = payload.error || JSON.stringify(payload, null, 2);
      runButton.disabled = false;
      return;
    }

    jobOutput.textContent = JSON.stringify(payload, null, 2);
    askButton.disabled = !(payload.status === 'complete' && payload.artifacts?.transcript_index);
    runButton.disabled = payload.status === 'complete';
  } catch (error) {
    jobOutput.textContent = error.message;
    runButton.disabled = currentJobId === null;
  }
});

document.querySelector('#question-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const body = {
    question: document.querySelector('#question').value,
    answer_language: document.querySelector('#answer-language').value
  };

  try {
    const payload = await fetchJson(`/api/jobs/${currentJobId}/questions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    answerOutput.textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    answerOutput.textContent = error.message;
  }
});
