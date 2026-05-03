let currentJobId = null;
let currentJob = null;
let isProcessing = false;

const jobForm = document.querySelector('#job-form');
const questionForm = document.querySelector('#question-form');
const jobOutput = document.querySelector('#job-output');
const runButton = document.querySelector('#run-job');
const askButton = document.querySelector('#ask-button');
const answerOutput = document.querySelector('#answer-output');
const statusPill = document.querySelector('#status-pill');
const jobSummary = document.querySelector('#job-summary');
const artifactList = document.querySelector('#artifact-list');
const answerCard = document.querySelector('#answer-card');
const createButton = document.querySelector('#create-job');

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

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function appendText(parent, tagName, text, className = '') {
  const child = document.createElement(tagName);
  child.textContent = text;
  if (className) {
    child.className = className;
  }
  parent.appendChild(child);
  return child;
}

function setStatus(label, tone = 'idle') {
  statusPill.textContent = label;
  statusPill.className = `status-pill ${tone}`;
}

function statusTone(status) {
  if (status === 'complete') return 'success';
  if (status === 'failed') return 'error';
  if (status === 'running' || status === 'processing') return 'active';
  if (status === 'queued') return 'queued';
  return 'idle';
}

function titleCase(value) {
  if (!value) return 'Unknown';
  return value.replace(/[_-]/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function jobOptions(job) {
  return job?.options || {};
}

function languageLabel(value) {
  if (value === 'he') return 'Hebrew';
  if (value === 'en') return 'English';
  if (value === 'auto') return 'Auto detect';
  return titleCase(value);
}

function canAsk(job) {
  return Boolean(job?.status === 'complete' && job?.artifacts?.transcript_index);
}

function syncButtons() {
  const hasJob = currentJobId !== null;
  createButton.disabled = isProcessing;
  runButton.disabled = isProcessing || !hasJob || currentJob?.status === 'complete';
  askButton.disabled = isProcessing || !canAsk(currentJob);
}

function renderEmptyJob(message = 'No job yet.') {
  clearNode(jobSummary);
  appendText(jobSummary, 'p', message, 'empty-state');
  jobOutput.hidden = true;
  renderArtifacts({});
  setStatus('Ready', 'idle');
  syncButtons();
}

function renderJob(job) {
  currentJob = job;
  currentJobId = job?.id ?? null;
  clearNode(jobSummary);

  if (!job) {
    renderEmptyJob();
    return;
  }

  const header = document.createElement('div');
  header.className = 'summary-header';
  appendText(header, 'strong', `Job ${job.id}`);
  appendText(header, 'span', titleCase(job.status), `mini-pill ${statusTone(job.status)}`);
  jobSummary.appendChild(header);

  const options = jobOptions(job);
  const grid = document.createElement('dl');
  grid.className = 'summary-grid';
  addSummaryItem(grid, 'Media file', job.input_path || 'Unknown');
  addSummaryItem(grid, 'Language', languageLabel(options.source_language || 'auto'));
  addSummaryItem(grid, 'Translate', options.translation_target ? languageLabel(options.translation_target) : 'None');
  addSummaryItem(grid, 'Subtitles', options.subtitles ? 'Yes' : 'No');
  addSummaryItem(grid, 'Face blur', options.face_blur ? 'Yes' : 'No');
  jobSummary.appendChild(grid);

  if (job.error) {
    appendText(jobSummary, 'p', job.error, 'error-message');
  }

  jobOutput.textContent = JSON.stringify(job, null, 2);
  jobOutput.hidden = true;
  renderArtifacts(job.artifacts || {});
  setStatus(titleCase(job.status), statusTone(job.status));
  syncButtons();
}

function addSummaryItem(parent, label, value) {
  const item = document.createElement('div');
  const term = document.createElement('dt');
  const description = document.createElement('dd');
  term.textContent = label;
  description.textContent = value;
  item.append(term, description);
  parent.appendChild(item);
}

function renderArtifacts(artifacts) {
  clearNode(artifactList);
  const entries = Object.entries(artifacts || {});

  if (entries.length === 0) {
    appendText(artifactList, 'li', 'No output files yet.', 'empty-state');
    return;
  }

  for (const [key, path] of entries) {
    const item = document.createElement('li');
    const label = appendText(item, 'span', artifactLabel(key), 'artifact-name');
    label.title = key;
    appendText(item, 'code', path, 'artifact-path');
    artifactList.appendChild(item);
  }
}

function artifactLabel(key) {
  const labels = {
    audio: 'Working audio',
    downloaded_media: 'Downloaded media',
    face_blurred_video: 'Blurred video',
    processing_report: 'Processing report',
    source_audio: 'Original audio',
    source_media: 'Original media',
    source_video: 'Original video',
    subtitles: 'Subtitles',
    transcript: 'Transcript',
    transcript_index: 'Transcript index',
    translated_subtitles: 'Translated subtitles',
    translated_transcript: 'Translated transcript'
  };
  return labels[key] || titleCase(key);
}

function renderError(target, message) {
  clearNode(target);
  appendText(target, 'p', message, 'error-message');
}

function secondsToClock(value) {
  if (!Number.isFinite(value)) return '00:00';
  const total = Math.max(0, Math.round(value));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function renderAnswer(payload) {
  clearNode(answerCard);

  appendText(answerCard, 'p', payload.answer || 'No answer found in the transcript.', 'answer-text');

  if (payload.confidence) {
    appendText(answerCard, 'p', `Confidence: ${titleCase(payload.confidence)}`, 'answer-meta');
  }

  const excerpts = payload.excerpts || [];
  if (excerpts.length > 0) {
    const list = document.createElement('ul');
    list.className = 'excerpt-list';
    for (const excerpt of excerpts) {
      const item = document.createElement('li');
      const range = `${secondsToClock(excerpt.start)}-${secondsToClock(excerpt.end)}`;
      appendText(item, 'span', range, 'excerpt-time');
      appendText(item, 'p', excerpt.text || '', 'excerpt-text');
      list.appendChild(item);
    }
    answerCard.appendChild(list);
  }

  answerOutput.textContent = JSON.stringify(payload, null, 2);
}

function setProcessingState(active, label = 'Processing') {
  isProcessing = active;
  if (active) {
    setStatus(label, 'active');
  } else if (currentJob) {
    setStatus(titleCase(currentJob.status), statusTone(currentJob.status));
  } else {
    setStatus('Ready', 'idle');
  }
  syncButtons();
}

jobForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const body = {
    input_path: document.querySelector('#input-path').value,
    source_language: document.querySelector('#source-language').value,
    translation_target: document.querySelector('#translation-target').value || null,
    subtitles: document.querySelector('#subtitles').checked,
    face_blur: document.querySelector('#face-blur').checked
  };

  try {
    setProcessingState(true, 'Creating');
    const payload = await fetchJson('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!payload.id) {
      throw new Error('Job response did not include an id.');
    }

    setProcessingState(false);
    renderJob(payload);
    clearNode(answerCard);
    appendText(answerCard, 'p', 'Process the media file first.', 'empty-state');
  } catch (error) {
    currentJobId = null;
    currentJob = null;
    setProcessingState(false);
    renderEmptyJob();
    renderError(jobSummary, error.message);
    syncButtons();
  }
});

runButton.addEventListener('click', async () => {
  if (currentJobId === null) return;

  setProcessingState(true);
  clearNode(answerCard);
  appendText(answerCard, 'p', 'Processing transcript...', 'empty-state');

  try {
    const payload = await fetchJson(`/api/jobs/${currentJobId}/run`, { method: 'POST' });
    setProcessingState(false);
    renderJob(payload);

    if (payload.status === 'failed') {
      renderError(answerCard, payload.error || 'Processing failed.');
      return;
    }

    if (canAsk(payload)) {
      clearNode(answerCard);
      appendText(answerCard, 'p', 'Transcript is ready for questions.', 'empty-state');
    }
  } catch (error) {
    setProcessingState(false);
    renderError(jobSummary, error.message);
    syncButtons();
  }
});

questionForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (currentJobId === null) return;

  const body = {
    question: document.querySelector('#question').value,
    answer_language: document.querySelector('#answer-language').value
  };

  try {
    askButton.disabled = true;
    clearNode(answerCard);
    appendText(answerCard, 'p', 'Finding the answer...', 'empty-state');
    const payload = await fetchJson(`/api/jobs/${currentJobId}/questions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    renderAnswer(payload);
  } catch (error) {
    renderError(answerCard, error.message);
  } finally {
    syncButtons();
  }
});

renderEmptyJob();
