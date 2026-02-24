/**
 * Session Sidebar Controller
 * Manages the session list UI with create/select/rename/delete functionality
 */

const Sidebar = {
    isCollapsed: false,
    editingSessionId: null,

    /**
     * Initialize sidebar
     */
    init() {
        this.bindEvents();
        this.loadCollapsedState();
        
        // Subscribe to store changes
        Store.subscribe((state) => this.render(state));
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Toggle sidebar (collapse button in sidebar header)
        document.getElementById('sidebar-collapse-btn')?.addEventListener('click', () => this.toggle());
        
        // Expand button in main header (visible when collapsed)
        document.getElementById('sidebar-expand-btn')?.addEventListener('click', () => this.toggle());
        
        // New chat button
        document.getElementById('new-chat-btn')?.addEventListener('click', () => this.createNewSession());
        
        // Export/Import sessions
        document.getElementById('export-sessions-btn')?.addEventListener('click', () => this.exportSessions());
        document.getElementById('import-sessions-btn')?.addEventListener('click', () => {
            document.getElementById('import-sessions-input')?.click();
        });
        document.getElementById('import-sessions-input')?.addEventListener('change', (e) => this.importSessions(e));
        
        // Click outside to cancel rename
        document.addEventListener('click', (e) => {
            if (this.editingSessionId && !e.target.closest('.session-item-editing')) {
                this.cancelRename();
            }
        });
        
        // Escape to cancel rename
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.editingSessionId) {
                this.cancelRename();
            }
        });
    },

    /**
     * Load collapsed state from localStorage
     */
    loadCollapsedState() {
        const collapsed = localStorage.getItem('sidebar_collapsed') === 'true';
        this.isCollapsed = collapsed;
        this.updateCollapsedUI();
    },

    /**
     * Toggle sidebar collapsed state
     */
    toggle() {
        this.isCollapsed = !this.isCollapsed;
        localStorage.setItem('sidebar_collapsed', this.isCollapsed);
        this.updateCollapsedUI();
    },

    /**
     * Update UI based on collapsed state
     */
    updateCollapsedUI() {
        const sidebar = document.getElementById('sidebar');
        const collapseBtn = document.getElementById('sidebar-collapse-btn');
        
        if (sidebar) {
            sidebar.classList.toggle('collapsed', this.isCollapsed);
        }
        
        // Update the collapse button icon
        if (collapseBtn) {
            const icon = collapseBtn.querySelector('i');
            if (icon) {
                // When collapsed, show expand icon (pointing right), when open show hamburger
                if (this.isCollapsed) {
                    icon.className = 'fas fa-chevron-right';
                } else {
                    icon.className = 'fas fa-bars';
                }
            }
        }
    },

    /**
     * Render the session list
     */
    render(state) {
        const sessionList = document.getElementById('session-list');
        if (!sessionList) return;
        
        const sessions = Store.getAllSessions();
        const activeId = state.activeSessionId;
        
        if (sessions.length === 0) {
            sessionList.innerHTML = `
                <div class="session-empty">
                    <p>No conversations yet</p>
                    <p class="text-sm text-muted">Click "New Chat" to start</p>
                </div>
            `;
            return;
        }
        
        // Group sessions by date
        const groups = this.groupSessionsByDate(sessions);
        
        let html = '';
        for (const [label, groupSessions] of Object.entries(groups)) {
            if (groupSessions.length === 0) continue;
            
            html += `<div class="session-group">
                <div class="session-group-label">${label}</div>
                ${groupSessions.map(session => this.renderSessionItem(session, activeId)).join('')}
            </div>`;
        }
        
        sessionList.innerHTML = html;
        
        // Re-bind session item events
        this.bindSessionItemEvents();
    },

    /**
     * Group sessions by date
     */
    groupSessionsByDate(sessions) {
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const yesterday = new Date(today.getTime() - 86400000);
        const weekAgo = new Date(today.getTime() - 7 * 86400000);
        
        const groups = {
            'Today': [],
            'Yesterday': [],
            'Previous 7 Days': [],
            'Older': []
        };
        
        for (const session of sessions) {
            const date = new Date(session.updatedAt);
            
            if (date >= today) {
                groups['Today'].push(session);
            } else if (date >= yesterday) {
                groups['Yesterday'].push(session);
            } else if (date >= weekAgo) {
                groups['Previous 7 Days'].push(session);
            } else {
                groups['Older'].push(session);
            }
        }
        
        return groups;
    },

    /**
     * Render a single session item
     */
    renderSessionItem(session, activeId) {
        const isActive = session.id === activeId;
        const isEditing = session.id === this.editingSessionId;
        const messageCount = session.messages.length;
        const preview = this.getSessionPreview(session);
        
        if (isEditing) {
            return `
                <div class="session-item session-item-editing active" data-id="${session.id}">
                    <input type="text" class="session-rename-input" value="${Utils.escapeHtml(session.title)}" 
                           onkeydown="Sidebar.handleRenameKeydown(event, '${session.id}')"
                           autofocus>
                    <div class="session-rename-actions">
                        <button class="session-action-btn" onclick="Sidebar.confirmRename('${session.id}')" title="Save">
                            <i class="fas fa-check"></i>
                        </button>
                        <button class="session-action-btn" onclick="Sidebar.cancelRename()" title="Cancel">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
            `;
        }
        
        return `
            <div class="session-item ${isActive ? 'active' : ''}" data-id="${session.id}">
                <div class="session-item-content" onclick="Sidebar.selectSession('${session.id}')">
                    <div class="session-title">${Utils.escapeHtml(session.title)}</div>
                    <div class="session-meta">
                        <span class="session-count">${messageCount} messages</span>
                    </div>
                </div>
                <div class="session-actions">
                    <button class="session-action-btn" onclick="event.stopPropagation(); Sidebar.startRename('${session.id}')" title="Rename">
                        <i class="fas fa-pen"></i>
                    </button>
                    <button class="session-action-btn session-action-delete" onclick="event.stopPropagation(); Sidebar.deleteSession('${session.id}')" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
    },

    /**
     * Get a preview of the session's content
     */
    getSessionPreview(session) {
        if (session.messages.length === 0) return 'New conversation';
        
        const lastMessage = session.messages[session.messages.length - 1];
        const content = lastMessage.content || '';
        return content.slice(0, 60) + (content.length > 60 ? '...' : '');
    },

    /**
     * Bind events to session items
     */
    bindSessionItemEvents() {
        // Focus rename input if editing
        if (this.editingSessionId) {
            const input = document.querySelector('.session-rename-input');
            if (input) {
                input.focus();
                input.select();
            }
        }
    },

    /**
     * Create a new session
     */
    createNewSession() {
        const session = Store.createSession();
        Chat.clearChat();
        Chat.showWelcome();
        Utils.showToast('New chat created', 'success');
    },

    /**
     * Select a session
     */
    selectSession(sessionId) {
        if (Store.selectSession(sessionId)) {
            Chat.loadSession(sessionId);
        }
    },

    /**
     * Start renaming a session
     */
    startRename(sessionId) {
        this.editingSessionId = sessionId;
        this.render(Store.state);
    },

    /**
     * Handle keydown in rename input
     */
    handleRenameKeydown(event, sessionId) {
        if (event.key === 'Enter') {
            event.preventDefault();
            this.confirmRename(sessionId);
        } else if (event.key === 'Escape') {
            this.cancelRename();
        }
    },

    /**
     * Confirm rename
     */
    confirmRename(sessionId) {
        const input = document.querySelector('.session-rename-input');
        const newTitle = input?.value.trim();
        
        if (newTitle && newTitle.length > 0) {
            Store.renameSession(sessionId, newTitle);
            Utils.showToast('Renamed', 'success');
        }
        
        this.editingSessionId = null;
        this.render(Store.state);
    },

    /**
     * Cancel rename
     */
    cancelRename() {
        this.editingSessionId = null;
        this.render(Store.state);
    },

    /**
     * Delete a session with confirmation
     */
    deleteSession(sessionId) {
        const session = Store.state.sessions[sessionId];
        if (!session) return;
        
        const messageCount = session.messages.length;
        const confirmMsg = messageCount > 0 
            ? `Delete "${session.title}" with ${messageCount} messages?`
            : `Delete "${session.title}"?`;
        
        if (confirm(confirmMsg)) {
            const wasActive = Store.state.activeSessionId === sessionId;
            Store.deleteSession(sessionId);
            
            if (wasActive) {
                const activeSession = Store.getActiveSession();
                if (activeSession) {
                    Chat.loadSession(activeSession.id);
                } else {
                    Chat.clearChat();
                    Chat.showWelcome();
                }
            }
            
            Utils.showToast('Deleted', 'success');
        }
    },

    /**
     * Export all sessions
     */
    exportSessions() {
        const json = Store.exportSessions();
        const filename = `chat-sessions-${new Date().toISOString().slice(0, 10)}.json`;
        Utils.downloadFile(json, filename);
        Utils.showToast('Sessions exported', 'success');
    },

    /**
     * Import sessions from file
     */
    async importSessions(event) {
        const file = event.target.files?.[0];
        if (!file) return;
        
        try {
            const text = await file.text();
            const result = Store.importSessions(text);
            
            if (result.success) {
                Utils.showToast(`Imported ${result.count} sessions`, 'success');
            } else {
                Utils.showToast(result.error, 'error');
            }
        } catch (e) {
            Utils.showToast('Failed to read file', 'error');
        }
        
        // Clear input
        event.target.value = '';
    }
};

// Export
window.Sidebar = Sidebar;
