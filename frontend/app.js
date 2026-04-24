const $ = (sel) => document.querySelector(sel);
const app = $("#app");
let currentPage = "gallery";
let state = {
    photos: [],
    categories: [],
    stats: null,
    currentCategory: null,
    galleryPage: 1,
    searchQuery: "",
    searchPage: 1,
    persons: [],
    duplicates: [],
    activeJobs: {},
};
const SEARCH_PER_PAGE = 40;

// --- API ---
async function api(path, opts = {}) {
    const res = await fetch(`/api${path}`, {
        headers: { "Content-Type": "application/json", ...opts.headers },
        ...opts,
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
}

// --- Router ---
function navigate(page) {
    currentPage = page;
    document.querySelectorAll(".nav-link").forEach((a) => {
        a.classList.toggle("active", a.dataset.page === page);
    });
    render();
}

window.addEventListener("hashchange", () => {
    const page = location.hash.slice(1) || "gallery";
    navigate(page);
});

document.querySelectorAll(".nav-link").forEach((a) => {
    a.addEventListener("click", (e) => {
        e.preventDefault();
        location.hash = a.dataset.page;
    });
});

// --- Toast ---
function toast(msg) {
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3000);
}

// --- Job Polling ---
function pollJob(jobId, onComplete) {
    const interval = setInterval(async () => {
        try {
            const job = await api(`/jobs/${jobId}`);
            state.activeJobs[jobId] = job;
            renderJobProgress();
            if (job.status === "completed" || job.status === "failed") {
                clearInterval(interval);
                delete state.activeJobs[jobId];
                if (job.status === "completed") {
                    toast(`${job.name} completed!`);
                    if (onComplete) onComplete(job);
                } else {
                    toast(`${job.name} failed`);
                }
                render();
            }
        } catch {
            clearInterval(interval);
        }
    }, 1000);
}

function renderJobProgress() {
    const el = document.getElementById("job-progress");
    if (!el) return;
    const jobs = Object.values(state.activeJobs);
    if (!jobs.length) {
        el.innerHTML = "";
        return;
    }
    el.innerHTML = jobs
        .map((j) => {
            const pct = j.total ? Math.round((j.progress / j.total) * 100) : 0;
            return `<div style="margin-bottom:0.5rem">
                <small>${j.name}: ${j.progress}/${j.total}</small>
                <div class="progress-bar"><div class="fill" style="width:${pct}%"></div></div>
            </div>`;
        })
        .join("");
}

// --- Modal ---
function closeModal() {
    $("#photo-modal").classList.remove("show");
}

async function openPhoto(photoId) {
    const photo = await api(`/photos/${photoId}`);
    const faces = await api(`/faces/${photoId}/crops`).catch(() => []);
    $("#modal-title").textContent = photo.filename;
    $("#modal-body").innerHTML = `
        <img src="/api/photos/${photoId}/full" alt="${photo.filename}">
        <div class="modal-info">
            <label>Date</label><p>${photo.taken_at || "Unknown"}</p>
            <label>Category</label><p>${photo.category || "Uncategorized"} ${photo.confidence ? `(${(photo.confidence * 100).toFixed(0)}%)` : ""}</p>
            <label>Size</label><p>${(photo.filesize / 1024 / 1024).toFixed(1)} MB</p>
            <label>Dimensions</label><p>${photo.width || "?"}x${photo.height || "?"}</p>
            <hr style="border-color:var(--border);margin:1rem 0">
            <label>Change Category</label>
            <select id="modal-category" style="margin-top:4px;padding:4px;background:var(--surface2);color:var(--text);border:1px solid var(--border);border-radius:4px">
                <option value="">-- Select --</option>
                ${state.categories.map((c) => `<option value="${c.name}" ${c.name === photo.category ? "selected" : ""}>${c.name}</option>`).join("")}
            </select>
            <button class="btn btn-sm btn-primary" style="margin-top:8px" onclick="setCategory(${photoId})">Save</button>
            ${faces.length ? `<hr style="border-color:var(--border);margin:1rem 0"><label>Faces detected: ${faces.length}</label>` : ""}
        </div>
    `;
    $("#photo-modal").classList.add("show");
}

async function setCategory(photoId) {
    const cat = document.getElementById("modal-category").value;
    if (!cat) return;
    await api(`/photos/${photoId}/category`, {
        method: "PUT",
        body: JSON.stringify({ category: cat }),
    });
    toast(`Category set to ${cat}`);
    closeModal();
    render();
}

$("#photo-modal").addEventListener("click", (e) => {
    if (e.target === $("#photo-modal")) closeModal();
});

// --- Pages ---
async function renderGallery() {
    const stats = await api("/stats");
    state.stats = stats;
    state.categories = await api("/categories");

    const params = new URLSearchParams({
        page: state.galleryPage,
        per_page: 60,
    });
    if (state.currentCategory) params.set("category", state.currentCategory);

    const data = await api(`/photos?${params}`);
    state.photos = data.photos;

    app.innerHTML = `
        <div class="stats">
            <div class="stat-card"><div class="number">${stats.total_photos}</div><div class="label">Photos</div></div>
            <div class="stat-card"><div class="number">${stats.classified}</div><div class="label">Classified</div></div>
            <div class="stat-card"><div class="number">${stats.persons}</div><div class="label">Persons</div></div>
            <div class="stat-card"><div class="number">${stats.duplicate_groups}</div><div class="label">Dup Groups</div></div>
        </div>
        <div id="job-progress"></div>
        <div class="toolbar">
            <button class="btn btn-primary" onclick="doScan()">Scan Photos</button>
            <button class="btn" onclick="doClassify()">Classify All</button>
            <button class="btn" onclick="doDetectFaces()">Detect Faces</button>
            <button class="btn" onclick="doScanDuplicates()">Find Duplicates</button>
        </div>
        <div class="pills">
            <span class="pill ${!state.currentCategory ? "active" : ""}" onclick="filterCategory(null)">All <span class="count">${stats.total_photos}</span></span>
            <span class="pill ${state.currentCategory === "Uncategorized" ? "active" : ""}" onclick="filterCategory('Uncategorized')">Uncategorized <span class="count">${stats.uncategorized}</span></span>
            ${state.categories.map((c) => `<span class="pill ${state.currentCategory === c.name ? "active" : ""}" onclick="filterCategory('${c.name}')">${c.name} <span class="count">${c.count}</span></span>`).join("")}
        </div>
        <div style="margin-top:1rem">
            ${data.photos.length ? `
                <div class="photo-grid">
                    ${data.photos.map((p) => `
                        <div class="photo-card" onclick="openPhoto(${p.id})">
                            <img src="/api/photos/${p.id}/thumbnail" loading="lazy" alt="${p.filename}">
                            <div class="label">${p.category || ""} ${p.taken_at ? "| " + p.taken_at.split("T")[0] : ""}</div>
                        </div>
                    `).join("")}
                </div>
                <div class="pagination">
                    ${state.galleryPage > 1 ? `<button class="btn btn-sm" onclick="changePage(${state.galleryPage - 1})">Prev</button>` : ""}
                    <span style="padding:0.25rem 0.5rem;color:var(--text-dim)">Page ${data.page} (${data.total} total)</span>
                    ${data.photos.length === 60 ? `<button class="btn btn-sm" onclick="changePage(${state.galleryPage + 1})">Next</button>` : ""}
                </div>
            ` : `<div class="empty"><h2>No photos yet</h2><p>Click "Scan Photos" to import your photo library</p></div>`}
        </div>
    `;
    renderJobProgress();
}

async function renderSearch() {
    app.innerHTML = `
        <h2 style="margin-bottom:1rem">Search Photos</h2>
        <div class="search-box">
            <input type="text" id="search-input" placeholder="Describe what you're looking for... (e.g. beach sunset, concert, food)" value="${state.searchQuery}">
            <button class="btn btn-primary" onclick="doSearch()">Search</button>
        </div>
        <div id="search-results"></div>
    `;
    document.getElementById("search-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") doSearch();
    });
}

async function renderFaces() {
    state.persons = await api("/faces/persons");
    app.innerHTML = `
        <h2 style="margin-bottom:1rem">People</h2>
        <div class="toolbar">
            <button class="btn btn-primary" onclick="doDetectFaces()">Detect Faces</button>
            <button class="btn" onclick="doClusterFaces()">Cluster Faces</button>
        </div>
        <div id="job-progress"></div>
        ${state.persons.length ? `
            <div class="face-grid">
                ${state.persons.map((p) => `
                    <div class="face-card" onclick="showPersonPhotos(${p.id})">
                        <div class="name">${p.name || "Person " + p.id}</div>
                        <div class="count">${p.photo_count} photos, ${p.face_count} faces</div>
                        <input type="text" placeholder="Name..." value="${p.name || ""}"
                               onclick="event.stopPropagation()"
                               onchange="namePerson(${p.id}, this.value)"
                               style="margin-top:8px;width:100%;padding:4px;background:var(--surface2);color:var(--text);border:1px solid var(--border);border-radius:4px;text-align:center">
                    </div>
                `).join("")}
            </div>
        ` : `<div class="empty"><h2>No faces detected yet</h2><p>Click "Detect Faces" then "Cluster Faces"</p></div>`}
    `;
    renderJobProgress();
}

async function renderDuplicates() {
    state.duplicates = await api("/duplicates");
    const anyKept = state.duplicates.some((g) => g.photos.some((p) => p.is_kept));
    app.innerHTML = `
        <h2 style="margin-bottom:1rem">Duplicates</h2>
        <div class="toolbar">
            <button class="btn btn-primary" onclick="doScanDuplicates()">Scan for Duplicates</button>
            ${anyKept ? `<button class="btn" onclick="doCleanupDuplicates()">Remove non-kept</button>` : ""}
        </div>
        <div id="job-progress"></div>
        ${state.duplicates.length ? state.duplicates.map((g) => `
            <div class="dup-group">
                <strong>Group ${g.group_id}</strong> (${g.photos.length} photos)
                <div class="dup-photos" style="margin-top:0.5rem">
                    ${g.photos.map((p) => `
                        <div class="dup-photo ${p.is_kept ? "kept" : ""}">
                            <img src="/api/photos/${p.id}/thumbnail" onclick="openPhoto(${p.id})">
                            <div style="margin-top:4px">
                                <small>${(p.filesize / 1024).toFixed(0)} KB</small><br>
                                <button class="btn btn-sm" onclick="keepDuplicate(${g.group_id}, ${p.id})">Keep</button>
                            </div>
                        </div>
                    `).join("")}
                </div>
            </div>
        `).join("") : `<div class="empty"><h2>No duplicates found</h2><p>Click "Scan for Duplicates" to check</p></div>`}
    `;
    renderJobProgress();
}

async function renderSettings() {
    const res = await fetch("/api/categories");
    const categories = await res.json();

    let configText = "";
    try {
        const configRes = await fetch("/api/health");
        configText = `# Edit your categories below and click Save\n# Each category needs a name and prompts list\n\ncategories:\n${categories.map((c) => `  - name: "${c.name}"\n    count: ${c.count}`).join("\n")}\n\n# To update categories, edit config/categories.yml directly\n# and restart the backend`;
    } catch { configText = "Could not load config"; }

    const stats = await api("/stats");
    app.innerHTML = `
        <h2 style="margin-bottom:1rem">Settings</h2>
        <div class="stats">
            <div class="stat-card"><div class="number">${stats.total_photos}</div><div class="label">Total Photos</div></div>
            <div class="stat-card"><div class="number">${stats.classified}</div><div class="label">Classified</div></div>
            <div class="stat-card"><div class="number">${stats.uncategorized}</div><div class="label">Uncategorized</div></div>
            <div class="stat-card"><div class="number">${stats.faces_detected}</div><div class="label">Faces</div></div>
            <div class="stat-card"><div class="number">${stats.persons}</div><div class="label">Persons</div></div>
            <div class="stat-card"><div class="number">${stats.duplicate_groups}</div><div class="label">Duplicate Groups</div></div>
        </div>
        <h3 style="margin:1.5rem 0 0.5rem">Current Categories</h3>
        <p style="color:var(--text-dim);margin-bottom:1rem">Edit <code>config/categories.yml</code> to change categories, then reclassify.</p>
        <textarea class="config-editor" readonly>${configText}</textarea>
        <div style="margin-top:1rem">
            <button class="btn btn-primary" onclick="doClassify(true)">Reclassify All Photos</button>
        </div>
    `;
}

// --- Actions ---
async function doScan() {
    const res = await api("/scan", { method: "POST" });
    if (res.job_id) {
        toast(`Scanning ${res.total} files...`);
        pollJob(res.job_id, () => render());
    } else {
        toast(res.message);
    }
}

async function doClassify(force = false) {
    const res = await api(`/classify${force ? "?force=true" : ""}`, { method: "POST" });
    if (res.job_id) {
        toast(`Classifying ${res.total} photos...`);
        pollJob(res.job_id, () => render());
    } else {
        toast(res.message);
    }
}

async function doDetectFaces() {
    const res = await api("/faces/detect", { method: "POST" });
    if (res.job_id) {
        toast(`Detecting faces in ${res.total} photos...`);
        pollJob(res.job_id, () => render());
    } else {
        toast(res.message);
    }
}

async function doClusterFaces() {
    const res = await api("/faces/cluster", { method: "POST" });
    toast(`Created ${res.clusters_created} clusters`);
    render();
}

async function doScanDuplicates() {
    const res = await api("/duplicates/scan", { method: "POST" });
    if (res.job_id) {
        toast("Scanning for duplicates...");
        pollJob(res.job_id, () => render());
    }
}

async function doSearch(resetPage = true) {
    const q = document.getElementById("search-input").value.trim();
    if (!q) return;
    if (resetPage) state.searchPage = 1;
    state.searchQuery = q;
    const data = await api(
        `/search?q=${encodeURIComponent(q)}&page=${state.searchPage}&per_page=${SEARCH_PER_PAGE}`
    );
    const el = document.getElementById("search-results");
    if (!data.results.length) {
        el.innerHTML = `<div class="empty"><p>No results for "${q}"</p></div>`;
        return;
    }
    const hasPrev = state.searchPage > 1;
    const hasNext = data.results.length === SEARCH_PER_PAGE;
    el.innerHTML = `
        <p style="color:var(--text-dim);margin-bottom:1rem">
            Showing ${data.results.length} results (${data.total} embedded photos searched)
        </p>
        <div class="photo-grid">
            ${data.results.map((r) => `
                <div class="photo-card" onclick="openPhoto(${r.photo.id})">
                    <img src="/api/photos/${r.photo.id}/thumbnail" loading="lazy">
                    <div class="label">${(r.score * 100).toFixed(0)}% match</div>
                </div>
            `).join("")}
        </div>
        <div class="pagination">
            ${hasPrev ? `<button class="btn btn-sm" onclick="changeSearchPage(${state.searchPage - 1})">Prev</button>` : ""}
            <span style="padding:0.25rem 0.5rem;color:var(--text-dim)">Page ${data.page}</span>
            ${hasNext ? `<button class="btn btn-sm" onclick="changeSearchPage(${state.searchPage + 1})">Next</button>` : ""}
        </div>
    `;
}

function changeSearchPage(p) {
    state.searchPage = p;
    doSearch(false);
}

function filterCategory(cat) {
    state.currentCategory = cat;
    state.galleryPage = 1;
    render();
}

function changePage(p) {
    state.galleryPage = p;
    render();
}

async function namePerson(personId, name) {
    await api(`/faces/persons/${personId}`, {
        method: "PUT",
        body: JSON.stringify({ name }),
    });
    toast(`Named as ${name}`);
}

async function showPersonPhotos(personId) {
    const data = await api(`/faces/persons/${personId}`);
    app.innerHTML = `
        <button class="btn" onclick="navigate('faces')" style="margin-bottom:1rem">&larr; Back to Faces</button>
        <h2>${data.person.name || "Person " + data.person.id}</h2>
        <div class="photo-grid" style="margin-top:1rem">
            ${data.photos.map((p) => `
                <div class="photo-card" onclick="openPhoto(${p.id})">
                    <img src="/api/photos/${p.id}/thumbnail" loading="lazy">
                </div>
            `).join("")}
        </div>
    `;
}

async function keepDuplicate(groupId, photoId) {
    await api(`/duplicates/${groupId}/keep/${photoId}`, { method: "POST" });
    toast("Marked as keep");
    render();
}

async function doCleanupDuplicates() {
    const preview = await api("/duplicates/cleanup?dry_run=true", { method: "POST" });
    if (!preview.would_delete) {
        toast("Nothing to delete — mark a photo as keep in each group first");
        return;
    }
    const msg = `Delete ${preview.would_delete} photo entries across ${preview.processed_groups} groups?\n\n`
              + `(${preview.skipped_groups} groups skipped — no keep marked)\n\n`
              + `Note: the original files on disk are NOT removed. Only the database entries and thumbnails.`;
    if (!confirm(msg)) return;
    const result = await api("/duplicates/cleanup?dry_run=false", { method: "POST" });
    toast(`Deleted ${result.deleted} entries`);
    render();
}

// --- Render ---
function render() {
    switch (currentPage) {
        case "gallery": renderGallery(); break;
        case "search": renderSearch(); break;
        case "faces": renderFaces(); break;
        case "duplicates": renderDuplicates(); break;
        case "settings": renderSettings(); break;
    }
}

// Init
const initialPage = location.hash.slice(1) || "gallery";
navigate(initialPage);
