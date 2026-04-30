let currentJobId = null;

const jobOutput = document.querySelector('#job-output');
const runButton = document.querySelector('#run-job');
const askButton = document.querySelector('#ask-button');
const answerOutput = document.querySelector('#answer-output');

document.querySelector('#job-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const body = {
    input_path: document.querySelector('#input-path').value,
    source_language: document.querySelector('#source-language').value,
    translation_target: document.querySelector('#translation-target').value || null,
    subtitles: document.querySelector('#subtitles').checked,
    face_blur: document.querySelector('#face-blur').checked
  };
  const response = await fetch('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const payload = await response.json();
  currentJobId = payload.id;
  jobOutput.textContent = JSON.stringify(payload, null, 2);
  runButton.disabled = false;
});

runButton.addEventListener('click', async () => {
  runButton.disabled = true;
  jobOutput.textContent = 'Processing...';
  const response = await fetch(`/api/jobs/${currentJobId}/run`, { method: 'POST' });
  const payload = await response.json();
  jobOutput.textContent = JSON.stringify(payload, null, 2);
  askButton.disabled = false;
});

document.querySelector('#question-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const body = {
    question: document.querySelector('#question').value,
    answer_language: document.querySelector('#answer-language').value
  };
  const response = await fetch(`/api/jobs/${currentJobId}/questions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const payload = await response.json();
  answerOutput.textContent = JSON.stringify(payload, null, 2);
});
