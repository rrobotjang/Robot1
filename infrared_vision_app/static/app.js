const TOKEN_STORAGE_KEY = "infrared-vision-token";
const REFRESH_TOKEN_STORAGE_KEY = "infrared-vision-refresh-token";
const PROFILE_STORAGE_KEY = "infrared-vision-profile";
const EXPIRES_AT_STORAGE_KEY = "infrared-vision-expires-at";
const AUTO_REFRESH_MS = 20000;
const REFRESH_BUFFER_MS = 60000;

const state = {
  profile: localStorage.getItem(PROFILE_STORAGE_KEY) || "logistics",
  token: localStorage.getItem(TOKEN_STORAGE_KEY) || "",
  refreshToken: localStorage.getItem(REFRESH_TOKEN_STORAGE_KEY) || "",
  expiresAt: localStorage.getItem(EXPIRES_AT_STORAGE_KEY) || "",
  user: null,
  dashboard: null,
  selectedEventId: null,
  inference: null,
  search: null,
  vectorStatus: null,
  systemStatus: null,
  auditLogs: [],
  lastSyncedAt: null,
  notice: {
    message: "운영 상태를 준비하고 있습니다.",
    tone: "info",
  },
};

let refreshPromise = null;
let pollingTimer = null;
let sessionRefreshTimer = null;

const loginOverlay = document.getElementById("login-overlay");
const loginForm = document.getElementById("login-form");
const loginError = document.getElementById("login-error");
const usernameInput = document.getElementById("username-input");
const passwordInput = document.getElementById("password-input");

const profileSwitch = document.getElementById("profile-switch");
const userChip = document.getElementById("user-chip");
const logoutButton = document.getElementById("logout-button");
const refreshButton = document.getElementById("refresh-button");
const summaryGrid = document.getElementById("summary-grid");
const cameraGrid = document.getElementById("camera-grid");
const eventList = document.getElementById("event-list");
const eventDetail = document.getElementById("event-detail");
const timeline = document.getElementById("timeline");
const heroLocation = document.getElementById("hero-location");
const heroTitle = document.getElementById("hero-title");
const heroSubtitle = document.getElementById("hero-subtitle");
const badgeRole = document.getElementById("badge-role");
const badgeVector = document.getElementById("badge-vector");
const noticeBanner = document.getElementById("notice-banner");
const syncChip = document.getElementById("sync-chip");

const mobileNav = document.getElementById("mobile-nav");
const inferenceForm = document.getElementById("inference-form");
const inferenceCamera = document.getElementById("inference-camera");
const inferenceSummary = document.getElementById("inference-summary");
const inferenceObjects = document.getElementById("inference-objects");
const inferenceSlidingWindow = document.getElementById("inference-sliding-window");
const inferenceThermalMatrix = document.getElementById("inference-thermal-matrix");
const inferenceNote = document.getElementById("inference-note");
const inferenceAutoCreate = document.getElementById("inference-auto-create");
const inferenceResult = document.getElementById("inference-result");
const inferenceLockNote = document.getElementById("inference-lock-note");

const searchForm = document.getElementById("search-form");
const searchQuery = document.getElementById("search-query");
const searchResults = document.getElementById("search-results");
const systemStatus = document.getElementById("system-status");
const auditLogs = document.getElementById("audit-logs");

function roleLabel(role) {
  const labels = {
    viewer: "조회자",
    operator: "운영자",
    admin: "관리자",
  };
  return labels[role] || role;
}

function statusLabel(status) {
  const labels = {
    new: "신규",
    acknowledged: "확인됨",
    in_progress: "조치중",
    resolved: "해결됨",
  };
  return labels[status] || status;
}

function riskLabel(level) {
  const labels = {
    low: "LOW",
    medium: "MEDIUM",
    high: "HIGH",
    critical: "CRITICAL",
  };
  return labels[level] || level;
}

function formatDate(value) {
  return new Intl.DateTimeFormat("ko-KR", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatDuration(seconds) {
  if (seconds < 60) {
    return `${seconds}초`;
  }
  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}분`;
  }
  return `${Math.floor(seconds / 3600)}시간`;
}

function setNotice(message, tone = "info") {
  state.notice = { message, tone };
}

function setStoredProfile(profile) {
  state.profile = profile;
  localStorage.setItem(PROFILE_STORAGE_KEY, profile);
}

function persistSession(payload) {
  state.token = payload.access_token;
  state.refreshToken = payload.refresh_token;
  state.expiresAt = payload.expires_at;
  state.user = payload.user;
  localStorage.setItem(TOKEN_STORAGE_KEY, payload.access_token);
  localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, payload.refresh_token);
  localStorage.setItem(EXPIRES_AT_STORAGE_KEY, payload.expires_at);
}

function clearSession() {
  state.token = "";
  state.refreshToken = "";
  state.expiresAt = "";
  state.user = null;
  localStorage.removeItem(TOKEN_STORAGE_KEY);
  localStorage.removeItem(REFRESH_TOKEN_STORAGE_KEY);
  localStorage.removeItem(EXPIRES_AT_STORAGE_KEY);

  if (pollingTimer) {
    window.clearInterval(pollingTimer);
    pollingTimer = null;
  }
  if (sessionRefreshTimer) {
    window.clearTimeout(sessionRefreshTimer);
    sessionRefreshTimer = null;
  }
}

function resetOperationalState() {
  state.dashboard = null;
  state.search = null;
  state.inference = null;
  state.vectorStatus = null;
  state.systemStatus = null;
  state.auditLogs = [];
  state.selectedEventId = null;
  state.lastSyncedAt = null;
}

function showLogin(message = "") {
  loginOverlay.classList.remove("is-hidden");
  loginError.textContent = message;
}

function hideLogin() {
  loginOverlay.classList.add("is-hidden");
  loginError.textContent = "";
}

function canOperate() {
  return state.user && ["operator", "admin"].includes(state.user.role);
}

function syncProfileWithAccess() {
  if (!state.user) {
    return;
  }

  const allowed = state.user.allowed_profiles || [];
  if (!allowed.includes(state.profile) && allowed.length > 0) {
    setStoredProfile(allowed[0]);
  }
}

function extractErrorMessage(body, response) {
  if (body && body.error && body.error.message) {
    return body.error.message;
  }
  if (body && body.detail) {
    return body.detail;
  }
  return `Request failed: ${response.status}`;
}

function parseThermalMatrix(raw) {
  if (!raw.trim()) {
    return [];
  }

  return raw
    .trim()
    .split(/\n+/)
    .map((line) =>
      line
        .trim()
        .split(/[\s,]+/)
        .filter(Boolean)
        .map((value) => Number.parseFloat(value))
    )
    .filter((row) => row.length > 0);
}

async function refreshSession() {
  if (!state.refreshToken) {
    throw new Error("세션을 갱신할 리프레시 토큰이 없습니다.");
  }

  if (refreshPromise) {
    return refreshPromise;
  }

  refreshPromise = (async () => {
    const response = await fetch("/api/auth/refresh", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        refresh_token: state.refreshToken,
      }),
    });

    let body = null;
    try {
      body = await response.json();
    } catch (error) {
      body = null;
    }

    if (!response.ok) {
      throw new Error(extractErrorMessage(body, response));
    }

    persistSession(body);
    syncProfileWithAccess();
    scheduleSessionRefresh();
    setNotice("세션을 안전하게 갱신했습니다.", "info");
    return body;
  })()
    .catch((error) => {
      clearSession();
      resetOperationalState();
      renderAll();
      showLogin("세션이 만료되었습니다. 다시 로그인해주세요.");
      throw error;
    })
    .finally(() => {
      refreshPromise = null;
    });

  return refreshPromise;
}

function scheduleSessionRefresh() {
  if (sessionRefreshTimer) {
    window.clearTimeout(sessionRefreshTimer);
    sessionRefreshTimer = null;
  }

  if (!state.expiresAt || !state.refreshToken) {
    return;
  }

  const delay = Math.max(new Date(state.expiresAt).getTime() - Date.now() - REFRESH_BUFFER_MS, 5000);
  sessionRefreshTimer = window.setTimeout(async () => {
    if (!document.hidden && state.user) {
      try {
        await refreshSession();
      } catch (error) {
        setNotice(error.message, "error");
      }
    }
  }, delay);
}

async function requestJson(path, options = {}, requiresAuth = true, allowRefresh = true) {
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };

  if (requiresAuth && state.token) {
    headers.Authorization = `Bearer ${state.token}`;
  }

  const response = await fetch(path, {
    ...options,
    headers,
  });

  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    body = null;
  }

  if (
    response.status === 401 &&
    requiresAuth &&
    allowRefresh &&
    state.refreshToken &&
    path !== "/api/auth/refresh" &&
    path !== "/api/auth/logout"
  ) {
    await refreshSession();
    return requestJson(path, options, requiresAuth, false);
  }

  if (response.status === 401) {
    clearSession();
    resetOperationalState();
    renderAll();
    showLogin("세션이 만료되었거나 로그인이 필요합니다.");
    throw new Error(extractErrorMessage(body, response));
  }

  if (!response.ok) {
    throw new Error(extractErrorMessage(body, response));
  }

  return body;
}

async function login(username, password) {
  const payload = await requestJson(
    "/api/auth/login",
    {
      method: "POST",
      body: JSON.stringify({ username, password }),
    },
    false,
    false
  );

  persistSession(payload);
  syncProfileWithAccess();
  scheduleSessionRefresh();
  setNotice("로그인되었습니다. 실시간 현장 데이터를 동기화합니다.", "info");
  hideLogin();
}

async function restoreSession() {
  if (!state.token && !state.refreshToken) {
    showLogin();
    return;
  }

  try {
    if (!state.token && state.refreshToken) {
      await refreshSession();
    }
    state.user = await requestJson("/api/auth/me");
    syncProfileWithAccess();
    scheduleSessionRefresh();
    hideLogin();
  } catch (error) {
    showLogin("로그인이 필요합니다.");
  }
}

async function loadVectorStatus() {
  state.vectorStatus = await requestJson("/api/vector/status");
}

async function loadSystemStatus() {
  state.systemStatus = await requestJson("/api/system/status");
}

async function loadAuditLogs() {
  if (!state.user || state.user.role !== "admin") {
    state.auditLogs = [];
    return;
  }
  state.auditLogs = await requestJson("/api/audit/logs?limit=12");
}

async function loadDashboard() {
  state.dashboard = await requestJson(`/api/dashboard?profile=${state.profile}`);
  const availableEventIds = state.dashboard.events.map((event) => event.id);
  if (!state.selectedEventId || !availableEventIds.includes(state.selectedEventId)) {
    state.selectedEventId = state.dashboard.selected_event_id;
  }
}

async function refreshAppData() {
  if (!state.user) {
    return;
  }

  refreshButton.disabled = true;
  refreshButton.textContent = "동기화 중";

  try {
    await Promise.all([loadDashboard(), loadVectorStatus(), loadSystemStatus(), loadAuditLogs()]);
    state.lastSyncedAt = new Date().toISOString();
    setNotice("현장 데이터 동기화가 완료되었습니다.", "success");
  } catch (error) {
    setNotice(error.message, "error");
    throw error;
  } finally {
    refreshButton.disabled = false;
    refreshButton.textContent = "새로고침";
  }

  renderAll();
}

function startPolling() {
  if (pollingTimer) {
    window.clearInterval(pollingTimer);
  }

  if (!state.user) {
    return;
  }

  pollingTimer = window.setInterval(async () => {
    if (document.hidden || !state.user) {
      return;
    }
    try {
      await refreshAppData();
    } catch (error) {
      setNotice(error.message, "error");
      renderAll();
    }
  }, AUTO_REFRESH_MS);
}

function renderUserChip() {
  if (!state.user) {
    userChip.textContent = "로그인 필요";
    badgeRole.textContent = "권한 미인증";
    [...profileSwitch.querySelectorAll("button")].forEach((button) => {
      button.disabled = true;
      button.classList.toggle("is-active", button.dataset.profile === state.profile);
    });
    return;
  }

  const profiles = state.user.allowed_profiles
    .map((profile) => (profile === "logistics" ? "물류" : "도로"))
    .join(" / ");
  const expiresCopy = state.expiresAt ? `${formatDate(state.expiresAt)} 만료` : "세션 정보 없음";

  userChip.textContent = `${state.user.full_name} · ${roleLabel(state.user.role)}`;
  badgeRole.textContent = `${roleLabel(state.user.role)} · ${profiles} · ${expiresCopy}`;

  [...profileSwitch.querySelectorAll("button")].forEach((button) => {
    button.disabled = !state.user.allowed_profiles.includes(button.dataset.profile);
    button.classList.toggle("is-active", button.dataset.profile === state.profile);
  });
}

function renderHero() {
  if (!state.dashboard) {
    heroLocation.textContent = "현장 로딩 중";
    heroTitle.textContent = "실시간 위험 상황을 불러오는 중입니다.";
    heroSubtitle.textContent = "카메라 상태와 이벤트 목록을 동기화하고 있습니다.";
    badgeVector.textContent = "Vector DB: -";
    return;
  }

  heroLocation.textContent = `${state.dashboard.site.name} · ${state.dashboard.site.location}`;
  heroTitle.textContent =
    state.dashboard.profile === "logistics"
      ? "물류센터의 작업자, 지게차, 제한구역 이벤트를 한 화면에서 대응합니다."
      : "저시야 도로의 보행자, 장애물, 공사구간 이벤트를 실시간으로 대응합니다.";
  heroSubtitle.textContent = `${formatDate(state.dashboard.generated_at)} 기준 상태입니다. 이벤트 대응과 벡터 가이드 검색이 함께 동작합니다.`;
  badgeVector.textContent = `Vector DB: ${
    state.vectorStatus ? state.vectorStatus.backend : "-"
  }`;
}

function renderStatusStrip() {
  noticeBanner.className = `notice-banner tone-${state.notice.tone}`;
  noticeBanner.textContent = state.notice.message;

  if (!state.user) {
    syncChip.textContent = "로그인 필요";
    return;
  }

  if (!state.lastSyncedAt) {
    syncChip.textContent = "동기화 대기";
    return;
  }

  const seconds = Math.max(0, Math.floor((Date.now() - new Date(state.lastSyncedAt).getTime()) / 1000));
  syncChip.textContent = `마지막 동기화 ${formatDuration(seconds)} 전`;
}

function renderSummary() {
  if (!state.dashboard) {
    summaryGrid.innerHTML = "";
    return;
  }

  summaryGrid.innerHTML = state.dashboard.summary
    .map(
      (card) => `
        <article class="summary-card ${card.tone}">
          <p class="eyebrow">${card.label}</p>
          <h3 class="value">${card.value}</h3>
          <p>${card.caption}</p>
        </article>
      `
    )
    .join("");
}

function renderCameras() {
  if (!state.dashboard) {
    cameraGrid.innerHTML = "";
    inferenceCamera.innerHTML = "";
    return;
  }

  cameraGrid.innerHTML = state.dashboard.cameras
    .map(
      (camera) => `
        <article class="camera-card">
          <div class="feed-preview">
            <p class="eyebrow">${camera.zone}</p>
            <h4>${camera.label}</h4>
            <small>${camera.stream_hint}</small>
          </div>
          <div class="camera-meta">
            <h4>${camera.id}</h4>
            <p>최근 감지 객체: ${camera.objects_detected.join(", ") || "없음"}</p>
            <p class="meta-copy">마지막 수신 ${formatDate(camera.last_seen_at)}</p>
            <span class="status-pill status-${camera.status}">
              ${camera.status.toUpperCase()}
            </span>
          </div>
        </article>
      `
    )
    .join("");

  inferenceCamera.innerHTML = state.dashboard.cameras
    .map((camera) => `<option value="${camera.id}">${camera.label} (${camera.id})</option>`)
    .join("");
}

function renderEvents() {
  if (!state.dashboard) {
    eventList.innerHTML = "";
    return;
  }

  eventList.innerHTML = state.dashboard.events
    .map(
      (event) => `
        <button class="event-row ${event.id === state.selectedEventId ? "is-selected" : ""}" data-select-event="${event.id}">
          <div class="event-row-top">
            <div>
              <p class="eyebrow">${event.zone}</p>
              <h4>${event.title}</h4>
            </div>
            <span class="risk-pill risk-${event.risk_level}">${riskLabel(event.risk_level)}</span>
          </div>
          <p>${event.summary}</p>
          <p class="meta-copy">${statusLabel(event.status)} · ${formatDate(event.occurred_at)}</p>
        </button>
      `
    )
    .join("");
}

function buildActionButtons(event) {
  if (!canOperate()) {
    return `<p class="lock-note">조회 권한 계정은 이벤트 상태를 변경할 수 없습니다.</p>`;
  }

  const buttons = [];
  if (event.status === "new") {
    buttons.push(`<button class="primary" type="button" data-ack-event="${event.id}">이벤트 확인</button>`);
  }
  if (event.status !== "in_progress" && event.status !== "resolved") {
    buttons.push(`<button type="button" data-status-event="${event.id}" data-status="in_progress">조치중 전환</button>`);
  }
  if (event.status !== "resolved") {
    buttons.push(`<button type="button" data-status-event="${event.id}" data-status="resolved">해결 완료</button>`);
  }
  return buttons.join("");
}

function renderSelectedEvent() {
  if (!state.dashboard) {
    eventDetail.className = "event-detail empty-state";
    eventDetail.textContent = "이벤트를 선택하면 상세 정보가 표시됩니다.";
    timeline.className = "timeline empty-state";
    timeline.textContent = "대응 이력이 아직 없습니다.";
    return;
  }

  const selected = state.dashboard.events.find((event) => event.id === state.selectedEventId);
  if (!selected) {
    eventDetail.className = "event-detail empty-state";
    eventDetail.textContent = "이벤트를 선택하면 상세 정보가 표시됩니다.";
    timeline.className = "timeline empty-state";
    timeline.textContent = "대응 이력이 아직 없습니다.";
    return;
  }

  eventDetail.className = "event-detail";
  const distanceRow =
    selected.distance_m !== null && selected.distance_m !== undefined
      ? `
        <div class="detail-gridline">
          <strong>
            <span>위험 거리</span>
            ${selected.distance_m.toFixed(1)}m
          </strong>
        </div>
      `
      : "";

  eventDetail.innerHTML = `
    <div class="detail-gridline">
      <strong>
        <span>이벤트 ID</span>
        ${selected.id}
      </strong>
      <span class="risk-pill risk-${selected.risk_level}">${riskLabel(selected.risk_level)}</span>
    </div>
    <div class="detail-gridline">
      <strong>
        <span>위험 유형</span>
        ${selected.risk_type}
      </strong>
      <strong>
        <span>객체 유형</span>
        ${selected.object_type}
      </strong>
    </div>
    <div class="detail-gridline">
      <strong>
        <span>발생 위치</span>
        ${selected.zone}
      </strong>
      <strong>
        <span>상태</span>
        ${statusLabel(selected.status)}
      </strong>
    </div>
    <div class="detail-gridline">
      <strong>
        <span>권장 조치</span>
        ${selected.recommended_action}
      </strong>
    </div>
    <div class="detail-gridline">
      <strong>
        <span>담당자</span>
        ${selected.assignee || "미지정"}
      </strong>
      <strong>
        <span>발생 시각</span>
        ${formatDate(selected.occurred_at)}
      </strong>
    </div>
    ${distanceRow}
    <div class="action-row">
      ${buildActionButtons(selected)}
      <button
        type="button"
        data-fill-search="${encodeURIComponent(
          `${selected.risk_type} ${selected.zone} ${selected.summary}`
        )}"
      >가이드 검색어 채우기</button>
    </div>
  `;

  timeline.className = "timeline";
  timeline.innerHTML = selected.timeline
    .slice()
    .reverse()
    .map(
      (entry) => `
        <article class="timeline-card">
          <p class="eyebrow">${formatDate(entry.at)}</p>
          <h4>${entry.title}</h4>
          <p>${entry.detail}</p>
        </article>
      `
    )
    .join("");
}

function renderInferencePanel() {
  const lockedMessage = canOperate()
    ? ""
    : "운영자 또는 관리자 계정에서만 추론 API를 실행할 수 있습니다.";
  inferenceLockNote.textContent = lockedMessage;

  [...inferenceForm.querySelectorAll("input, textarea, select, button")].forEach((element) => {
    element.disabled = !canOperate();
  });

  if (!state.inference) {
    inferenceResult.className = "result-panel empty-state";
    inferenceResult.textContent = "추론을 실행하면 감지 결과와 자동 생성 이벤트 후보가 표시됩니다.";
    return;
  }

  const detectionTiles = state.inference.detections
    .map(
      (detection) => `
        <div class="detection-tile">
          <strong>${detection.label}</strong>
          <p>confidence ${Math.round(detection.confidence * 100)}%</p>
          <p>heat ${Math.round(detection.heat_score * 100)}%</p>
        </div>
      `
    )
    .join("");

  const guides = state.inference.response_guides
    .map((guide) => `<span class="tag-pill">${guide.title}</span>`)
    .join("");

  const createdEventAction = state.inference.created_event
    ? `<button type="button" data-select-event="${state.inference.created_event.id}">생성 이벤트 보기</button>`
    : "";

  inferenceResult.className = "result-panel";
  inferenceResult.innerHTML = `
    <article class="inference-card">
      <div class="inference-headline">
        <div>
          <p class="eyebrow">${state.inference.camera.zone}</p>
          <h4>${state.inference.risk_type}</h4>
        </div>
        <span class="risk-pill risk-${state.inference.risk_level}">
          ${riskLabel(state.inference.risk_level)}
        </span>
      </div>
      <p>${state.inference.summary}</p>
      <p>권장 조치: ${state.inference.recommended_action}</p>
      <p>Detection Strategy: ${state.inference.detection_strategy}</p>
      <p>Scanned Windows: ${state.inference.scanned_windows} · Hotspots: ${state.inference.hotspot_count}</p>
      <p>Vector Backend: ${state.inference.vector_backend}</p>
      <div class="detection-grid">${detectionTiles}</div>
      <div>${guides}</div>
      <div class="action-row">
        ${createdEventAction}
      </div>
    </article>
  `;
}

function renderSearchResults() {
  if (!state.search) {
    searchResults.className = "search-results empty-state";
    searchResults.textContent = "검색 결과가 여기 표시됩니다.";
    return;
  }

  if (state.search.results.length === 0) {
    searchResults.className = "search-results empty-state";
    searchResults.textContent = "조건에 맞는 대응 가이드를 찾지 못했습니다.";
    return;
  }

  searchResults.className = "search-results";
  searchResults.innerHTML = state.search.results
    .map(
      (result) => `
        <article class="search-card">
          <p class="eyebrow">score ${result.score.toFixed(3)}</p>
          <h4>${result.title}</h4>
          <p>${result.snippet}</p>
          <div>
            ${result.tags.map((tag) => `<span class="tag-pill">${tag}</span>`).join("")}
          </div>
        </article>
      `
    )
    .join("");
}

function renderOpsPanels() {
  if (!state.systemStatus) {
    systemStatus.className = "empty-state";
    systemStatus.textContent = "시스템 상태를 불러오는 중입니다.";
  } else {
    systemStatus.className = "";
    systemStatus.innerHTML = `
      <article class="health-card">
        <p class="eyebrow">Runtime</p>
        <h4>${state.systemStatus.status.toUpperCase()}</h4>
        <p class="meta-copy">${state.systemStatus.app_version} · ${state.systemStatus.release_channel}</p>
        <div class="health-grid">
          <div class="health-metric">
            <strong>DB</strong>
            <p>${state.systemStatus.database_path}</p>
          </div>
          <div class="health-metric">
            <strong>Event Count</strong>
            <p>${state.systemStatus.event_count}</p>
          </div>
          <div class="health-metric">
            <strong>Audit Count</strong>
            <p>${state.systemStatus.audit_count}</p>
          </div>
          <div class="health-metric">
            <strong>Active Sessions</strong>
            <p>${state.systemStatus.active_sessions}</p>
          </div>
          <div class="health-metric">
            <strong>Vector Backend</strong>
            <p>${state.systemStatus.vector_backend}</p>
          </div>
          <div class="health-metric">
            <strong>Uptime</strong>
            <p>${formatDuration(state.systemStatus.uptime_seconds)}</p>
          </div>
        </div>
      </article>
    `;
  }

  if (!state.user || state.user.role !== "admin") {
    auditLogs.className = "empty-state";
    auditLogs.textContent = "감사 로그는 관리자 계정에서만 확인할 수 있습니다.";
    return;
  }

  if (!state.auditLogs || state.auditLogs.length === 0) {
    auditLogs.className = "empty-state";
    auditLogs.textContent = "감사 로그가 아직 없습니다.";
    return;
  }

  auditLogs.className = "audit-logs";
  auditLogs.innerHTML = state.auditLogs
    .map(
      (entry) => `
        <article class="timeline-card">
          <p class="eyebrow">${formatDate(entry.created_at)}</p>
          <h4>${entry.action}</h4>
          <p>${entry.actor}</p>
          <p class="meta-copy">${Object.entries(entry.payload)
            .map(([key, value]) => `${key}: ${value}`)
            .join(" · ")}</p>
        </article>
      `
    )
    .join("");
}

function renderAll() {
  renderUserChip();
  renderHero();
  renderStatusStrip();
  renderSummary();
  renderCameras();
  renderEvents();
  renderSelectedEvent();
  renderInferencePanel();
  renderSearchResults();
  renderOpsPanels();
}

async function handleLoginSubmit(event) {
  event.preventDefault();
  try {
    await login(usernameInput.value.trim(), passwordInput.value);
    passwordInput.value = "";
    await refreshAppData();
    startPolling();
  } catch (error) {
    loginError.textContent = error.message;
  }
}

async function handleProfileSwitch(event) {
  const button = event.target.closest("[data-profile]");
  if (!button || button.disabled) {
    return;
  }

  setStoredProfile(button.dataset.profile);
  await refreshAppData();
}

async function handleEventActions(event) {
  const selectButton = event.target.closest("[data-select-event]");
  if (selectButton) {
    state.selectedEventId = selectButton.dataset.selectEvent;
    renderAll();
    return;
  }

  const ackButton = event.target.closest("[data-ack-event]");
  if (ackButton) {
    await requestJson(`/api/events/${ackButton.dataset.ackEvent}/acknowledge`, {
      method: "POST",
    });
    await refreshAppData();
    return;
  }

  const statusButton = event.target.closest("[data-status-event]");
  if (statusButton) {
    await requestJson(`/api/events/${statusButton.dataset.statusEvent}/status`, {
      method: "POST",
      body: JSON.stringify({
        status: statusButton.dataset.status,
      }),
    });
    await refreshAppData();
    return;
  }

  const fillSearchButton = event.target.closest("[data-fill-search]");
  if (fillSearchButton) {
    searchQuery.value = decodeURIComponent(fillSearchButton.dataset.fillSearch);
    searchQuery.focus();
  }
}

async function handleInferenceSubmit(event) {
  event.preventDefault();
  if (!canOperate()) {
    return;
  }

  const objects = inferenceObjects.value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const thermalMatrix = parseThermalMatrix(inferenceThermalMatrix.value);

  state.inference = await requestJson("/api/inference/run", {
    method: "POST",
    body: JSON.stringify({
      profile: state.profile,
      camera_id: inferenceCamera.value,
      thermal_summary: inferenceSummary.value.trim(),
      detected_objects: objects,
      thermal_matrix: thermalMatrix,
      use_sliding_window: inferenceSlidingWindow.checked,
      operator_note: inferenceNote.value.trim() || null,
      auto_create_event: inferenceAutoCreate.checked,
    }),
  });

  if (state.inference.created_event) {
    state.selectedEventId = state.inference.created_event.id;
  }

  setNotice("추론 실행이 완료되었습니다. 생성 이벤트를 확인하세요.", "success");
  await refreshAppData();
  renderInferencePanel();
}

async function handleSearchSubmit(event) {
  event.preventDefault();
  const query = searchQuery.value.trim();
  if (!query) {
    return;
  }

  state.search = await requestJson("/api/vector/search", {
    method: "POST",
    body: JSON.stringify({
      query,
      profile: state.profile,
      limit: 4,
    }),
  });
  setNotice("벡터 가이드 검색이 완료되었습니다.", "success");
  renderSearchResults();
  renderStatusStrip();
}

function handleMobileNav(event) {
  const button = event.target.closest("[data-scroll-target]");
  if (!button) {
    return;
  }

  const target = document.getElementById(button.dataset.scrollTarget);
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function handleManualRefresh() {
  try {
    await refreshAppData();
  } catch (error) {
    setNotice(error.message, "error");
    renderStatusStrip();
  }
}

async function handleLogout() {
  try {
    if (state.token) {
      await requestJson(
        "/api/auth/logout",
        {
          method: "POST",
        },
        true,
        false
      );
    }
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    clearSession();
    resetOperationalState();
    renderAll();
    showLogin("로그아웃되었습니다.");
  }
}

async function handleVisibilityChange() {
  if (document.hidden) {
    return;
  }
  if (state.user) {
    try {
      await refreshAppData();
    } catch (error) {
      setNotice(error.message, "error");
      renderAll();
    }
  }
}

async function bootstrap() {
  loginForm.addEventListener("submit", handleLoginSubmit);
  profileSwitch.addEventListener("click", handleProfileSwitch);
  eventList.addEventListener("click", handleEventActions);
  eventDetail.addEventListener("click", handleEventActions);
  inferenceResult.addEventListener("click", handleEventActions);
  inferenceForm.addEventListener("submit", handleInferenceSubmit);
  searchForm.addEventListener("submit", handleSearchSubmit);
  mobileNav.addEventListener("click", handleMobileNav);
  logoutButton.addEventListener("click", handleLogout);
  refreshButton.addEventListener("click", handleManualRefresh);
  document.addEventListener("visibilitychange", handleVisibilityChange);

  await restoreSession();
  if (state.user) {
    await refreshAppData();
    startPolling();
  } else {
    renderAll();
  }
}

bootstrap().catch((error) => {
  setNotice(error.message, "error");
  heroTitle.textContent = "앱 초기화에 실패했습니다.";
  heroSubtitle.textContent = error.message;
  renderStatusStrip();
});
