from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from views.layout import render_layout

router = APIRouter(tags=["Views"])


@router.get("/profiles-ui", response_class=HTMLResponse)
@router.get("/profiles", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def get_profiles_ui_page():
    content = """
    <style>
        /* Warning Banner */
        .warning-banner {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.4);
            border-left: 5px solid #f59e0b;
            border-radius: 12px;
            padding: 16px 20px;
            display: flex;
            align-items: flex-start;
            gap: 14px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }
        .warning-icon {
            font-size: 1.5rem;
        }
        .warning-text h4 {
            color: #fbbf24;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .warning-text p {
            color: #cbd5e1;
            font-size: 0.88rem;
            line-height: 1.4;
        }

        /* Header Action Bar */
        .action-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 4px;
        }
        .search-box {
            flex: 1;
            min-width: 280px;
            position: relative;
        }
        .search-box input {
            width: 100%;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 12px 16px;
            color: #f8fafc;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s ease;
        }
        .search-box input:focus {
            border-color: #38bdf8;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
        }
        .btn-primary {
            background: #0284c7;
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 12px 22px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3);
        }
        .btn-primary:hover {
            background: #0369a1;
            transform: translateY(-1px);
        }

        /* Person Cards Grid */
        .profiles-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 20px;
            margin-top: 4px;
        }
        .card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .card:hover {
            border-color: #475569;
            transform: translateY(-2px);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .person-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .person-id-badge {
            background: #0f172a;
            border: 1px solid #334155;
            color: #38bdf8;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .card-meta {
            display: flex;
            gap: 16px;
            font-size: 0.82rem;
            color: #94a3b8;
        }
        .meta-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Thumbnail Grid */
        .thumb-list {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding-bottom: 6px;
            scrollbar-width: thin;
        }
        .thumb-item {
            width: 80px;
            height: 80px;
            border-radius: 10px;
            object-fit: cover;
            border: 1px solid #334155;
            flex-shrink: 0;
            background: #0f172a;
        }
        .no-images {
            color: #64748b;
            font-size: 0.85rem;
            font-style: italic;
            padding: 20px 0;
            text-align: center;
            width: 100%;
        }

        .card-actions {
            display: flex;
            gap: 10px;
            margin-top: 6px;
        }
        .btn-outline {
            flex: 1;
            background: transparent;
            border: 1px solid #334155;
            color: #38bdf8;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-outline:hover {
            background: #0f172a;
            border-color: #38bdf8;
        }
        .btn-danger {
            background: transparent;
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #ef4444;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-danger:hover {
            background: rgba(239, 68, 68, 0.15);
            border-color: #ef4444;
        }

        /* Toast Feedback */
        .toast {
            min-width: 320px;
            max-width: 450px;
            padding: 14px 18px;
            border-radius: 12px;
            font-size: 0.9rem;
            font-weight: 500;
            color: #ffffff;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            gap: 12px;
            animation: slideIn 0.3s ease;
        }
        .toast.success { background: #059669; border: 1px solid #10b981; }
        .toast.info { background: #d97706; border: 1px solid #f59e0b; }
        .toast.error { background: #dc2626; border: 1px solid #ef4444; }
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* Modal Dialog */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(11, 15, 25, 0.85);
            backdrop-filter: blur(8px);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
        }
        .modal-overlay.open {
            opacity: 1;
            pointer-events: auto;
        }
        .modal {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            width: 90%;
            max-width: 500px;
            padding: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
            display: flex;
            flex-direction: column;
            gap: 18px;
        }
        .modal-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #94a3b8;
        }
        .form-group input {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px 14px;
            color: #f8fafc;
            font-size: 0.95rem;
            outline: none;
        }
        .form-group input:focus {
            border-color: #38bdf8;
        }
        .modal-buttons {
            display: flex;
            justify-content: flex-end;
            gap: 12px;
            margin-top: 8px;
        }
        .btn-secondary {
            background: #334155;
            color: #cbd5e1;
            border: none;
            border-radius: 8px;
            padding: 10px 18px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
        }
    </style>

    <!-- Warning Banner -->
    <div class="warning-banner">
        <div class="warning-icon">⚠️</div>
        <div class="warning-text">
            <h4>Strict Face Photo Enrollment Rules</h4>
            <p>While uploading images, please ensure that <strong>exactly ONE person</strong> is present in the photo frame and all uploaded photos belong to the <strong>same target user</strong>. The system will automatically reject photos containing 0 faces or multiple faces.</p>
        </div>
    </div>

    <!-- Action Bar -->
    <div class="action-bar">
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Search person by name or profile ID..." onkeyup="filterProfiles()" />
        </div>
        <button class="btn-primary" onclick="openAddPersonModal()">
            <span>➕ Add New Person</span>
        </button>
    </div>

    <!-- Profiles Grid -->
    <div class="profiles-grid" id="profilesGrid">
        <div style="color: #64748b; font-size: 0.95rem;">Loading enrolled profiles...</div>
    </div>

    <!-- Add Person Modal -->
    <div class="modal-overlay" id="addPersonModal">
        <div class="modal">
            <div class="modal-title">Add New Person Profile</div>
            <div class="form-group">
                <label for="newProfileId">Profile ID (Unique Username/ID)</label>
                <input type="text" id="newProfileId" placeholder="e.g. shravan" />
            </div>
            <div class="form-group">
                <label for="newDisplayName">Full Display Name</label>
                <input type="text" id="newDisplayName" placeholder="e.g. Shravan" />
            </div>
            <div class="form-group">
                <label for="newPhotoFile">Face Photo (Must contain 1 person)</label>
                <input type="file" id="newPhotoFile" accept="image/*" />
            </div>
            <div class="modal-buttons">
                <button class="btn-secondary" onclick="closeModal('addPersonModal')">Cancel</button>
                <button class="btn-primary" onclick="submitAddPerson()">Save & Enroll</button>
            </div>
        </div>
    </div>

    <!-- Upload Photo Modal -->
    <div class="modal-overlay" id="uploadPhotoModal">
        <div class="modal">
            <div class="modal-title" id="uploadModalTitle">Upload Photo for Person</div>
            <input type="hidden" id="uploadTargetId" />
            <input type="hidden" id="uploadTargetName" />
            <div class="form-group">
                <label for="uploadPhotoFiles">Select Face Photos (1 person per frame)</label>
                <input type="file" id="uploadPhotoFiles" accept="image/*" multiple />
            </div>
            <div class="modal-buttons">
                <button class="btn-secondary" onclick="closeModal('uploadPhotoModal')">Cancel</button>
                <button class="btn-primary" onclick="submitUploadPhotos()">Upload & Enroll</button>
            </div>
        </div>
    </div>

    <script>
        let allProfiles = [];

        async function fetchProfiles() {
            try {
                const res = await fetch('/api/profiles');
                const data = await res.json();
                if (data.status === 'success') {
                    allProfiles = data.profiles;
                    renderProfiles(allProfiles);
                }
            } catch (e) {
                showToast('Failed to fetch profiles', 'error');
            }
        }

        function renderProfiles(profiles) {
            const grid = document.getElementById('profilesGrid');
            if (!profiles || profiles.length === 0) {
                grid.innerHTML = '<div style="color: #64748b; font-size: 0.95rem; grid-column: 1/-1;">No enrolled profiles found. Click "+ Add New Person" above to create one.</div>';
                return;
            }

            grid.innerHTML = profiles.map(p => {
                const imagesHtml = (p.images && p.images.length > 0)
                    ? p.images.map(img => `<img src="${img}" class="thumb-item" alt="${p.name}" />`).join('')
                    : '<div class="no-images">No photos uploaded yet</div>';

                return `
                    <div class="card" data-name="${p.name.toLowerCase()}" data-id="${p.profile_id.toLowerCase()}">
                        <div class="card-header">
                            <div class="person-title">${escapeHtml(p.name)}</div>
                            <div class="person-id-badge">ID: ${escapeHtml(p.profile_id)}</div>
                        </div>
                        <div class="card-meta">
                            <div class="meta-item">🧠 <strong>${p.sample_count}</strong> Enrolled Vector(s)</div>
                            <div class="meta-item">🖼️ <strong>${p.img_count}</strong> Photo(s)</div>
                        </div>
                        <div class="thumb-list">
                            ${imagesHtml}
                        </div>
                        <div class="card-actions">
                            <button class="btn-outline" onclick="openUploadPhotoModal('${escapeHtml(p.profile_id)}', '${escapeHtml(p.name)}')">📷 Add Photo</button>
                            <button class="btn-danger" onclick="deleteProfile('${escapeHtml(p.profile_id)}')">🗑️ Delete</button>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function filterProfiles() {
            const query = document.getElementById('searchInput').value.toLowerCase().trim();
            const filtered = allProfiles.filter(p => 
                p.name.toLowerCase().includes(query) || p.profile_id.toLowerCase().includes(query)
            );
            renderProfiles(filtered);
        }

        function openAddPersonModal() {
            document.getElementById('newProfileId').value = '';
            document.getElementById('newDisplayName').value = '';
            document.getElementById('newPhotoFile').value = '';
            document.getElementById('addPersonModal').classList.add('open');
        }

        function openUploadPhotoModal(profileId, name) {
            document.getElementById('uploadTargetId').value = profileId;
            document.getElementById('uploadTargetName').value = name;
            document.getElementById('uploadModalTitle').textContent = `Upload Photo for ${name}`;
            document.getElementById('uploadPhotoFiles').value = '';
            document.getElementById('uploadPhotoModal').classList.add('open');
        }

        function closeModal(modalId) {
            document.getElementById(modalId).classList.remove('open');
        }

        async function submitAddPerson() {
            const profileId = document.getElementById('newProfileId').value.trim();
            const name = document.getElementById('newDisplayName').value.trim();
            const fileInput = document.getElementById('newPhotoFile');

            if (!profileId) {
                showToast('Profile ID is required!', 'error');
                return;
            }

            if (fileInput.files.length > 0) {
                const formData = new FormData();
                formData.append('profile_id', profileId);
                formData.append('name', name || profileId);
                formData.append('file', fileInput.files[0]);

                await uploadSingleFile(formData);
            } else {
                const formData = new FormData();
                formData.append('profile_id', profileId);
                formData.append('name', name || profileId);

                try {
                    const res = await fetch('/api/profiles/create', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (res.ok) {
                        showToast(`Profile '${name || profileId}' created successfully!`, 'success');
                    } else {
                        showToast(data.detail || 'Failed to create profile', 'error');
                    }
                } catch (e) {
                    showToast('Server error while creating profile', 'error');
                }
            }

            closeModal('addPersonModal');
            fetchProfiles();
        }

        async function submitUploadPhotos() {
            const profileId = document.getElementById('uploadTargetId').value;
            const name = document.getElementById('uploadTargetName').value;
            const fileInput = document.getElementById('uploadPhotoFiles');

            if (!fileInput.files || fileInput.files.length === 0) {
                showToast('Please select at least one photo!', 'error');
                return;
            }

            closeModal('uploadPhotoModal');

            for (let i = 0; i < fileInput.files.length; i++) {
                const formData = new FormData();
                formData.append('profile_id', profileId);
                formData.append('name', name);
                formData.append('file', fileInput.files[i]);
                await uploadSingleFile(formData);
            }

            fetchProfiles();
        }

        async function uploadSingleFile(formData) {
            try {
                const res = await fetch('/api/profiles/enroll-image', { method: 'POST', body: formData });
                const data = await res.json();

                if (res.ok) {
                    if (data.code === 'ENROLLED') {
                        showToast(data.message, 'success');
                    } else if (data.code === 'DUPLICATE_SKIPPED') {
                        showToast(data.message, 'info');
                    }
                } else {
                    const err = data.detail || {};
                    showToast(err.message || 'Image enrollment failed', 'error');
                }
            } catch (e) {
                showToast('Failed to upload image', 'error');
            }
        }

        async function deleteProfile(profileId) {
            if (!confirm(`Are you sure you want to delete profile '${profileId}' and all its enrolled vectors?`)) return;

            try {
                const res = await fetch(`/api/profiles/${encodeURIComponent(profileId)}`, { method: 'DELETE' });
                const data = await res.json();
                if (res.ok) {
                    showToast(`Profile '${profileId}' deleted successfully!`, 'success');
                    fetchProfiles();
                } else {
                    showToast(data.detail || 'Failed to delete profile', 'error');
                }
            } catch (e) {
                showToast('Server error while deleting profile', 'error');
            }
        }

        function showToast(msg, type = 'info') {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.innerHTML = `<span>${msg}</span>`;
            container.appendChild(toast);

            setTimeout(() => {
                toast.remove();
            }, 4500);
        }

        function escapeHtml(str) {
            return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }

        fetchProfiles();
    </script>
    """
    html_page = render_layout("Person & Face Image Enrollment", "profiles", content)
    return HTMLResponse(content=html_page)
