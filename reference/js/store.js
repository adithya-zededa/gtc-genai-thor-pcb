/**
 * State Management Store
 * Centralized state management with reducers pattern
 * Four independent state domains: sessions, activeSessionId, llmConfig, uiState
 */

const Store = {
    // Storage keys
    STORAGE_KEYS: {
        SESSIONS: 'chat_sessions',
        ACTIVE_SESSION: 'chat_active_session',
        LLM_CONFIG: 'chat_llm_config',
        UI_STATE: 'chat_ui_state'
    },

    // State
    state: {
        sessions: {},           // Record<sessionId, Session>
        activeSessionId: null,  // string | null
        llmConfig: {
            provider: '',
            model: '',
            endpoint: '',
            parameters: {}
        },
        uiState: {
            isStreaming: false,
            activeAttachmentDraft: [],
            autoScrollEnabled: true
        }
    },

    // Subscribers for state changes
    listeners: new Set(),

    /**
     * Initialize store - load from localStorage
     */
    init() {
        this.loadFromStorage();
        this.handleInterruptedMessages();
        this.setupBeforeUnload();
        
        // Notify listeners of initial state
        this.notify();
    },

    /**
     * Subscribe to state changes
     */
    subscribe(listener) {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    },

    /**
     * Notify all listeners of state change
     */
    notify() {
        this.listeners.forEach(listener => {
            try {
                listener(this.state);
            } catch (e) {
                console.error('Store listener error:', e);
            }
        });
    },

    /**
     * Load all state from localStorage
     */
    loadFromStorage() {
        try {
            // Load sessions
            const sessionsJson = localStorage.getItem(this.STORAGE_KEYS.SESSIONS);
            if (sessionsJson) {
                const parsed = JSON.parse(sessionsJson);
                // Validate each session
                this.state.sessions = this.validateSessions(parsed);
            }

            // Load active session ID
            const activeId = localStorage.getItem(this.STORAGE_KEYS.ACTIVE_SESSION);
            if (activeId && this.state.sessions[activeId]) {
                this.state.activeSessionId = activeId;
            } else if (Object.keys(this.state.sessions).length > 0) {
                // Select most recent session
                const sorted = Object.values(this.state.sessions)
                    .sort((a, b) => b.updatedAt - a.updatedAt);
                this.state.activeSessionId = sorted[0]?.id || null;
            }

            // Load LLM config (global)
            const llmConfigJson = localStorage.getItem(this.STORAGE_KEYS.LLM_CONFIG);
            if (llmConfigJson) {
                const parsed = JSON.parse(llmConfigJson);
                this.state.llmConfig = this.validateLLMConfig(parsed);
            }

            // UI state is transient, don't load from storage
            
        } catch (e) {
            console.error('Failed to load from storage:', e);
            // Reset to defaults on error
            this.state.sessions = {};
            this.state.activeSessionId = null;
        }
    },

    /**
     * Validate sessions data structure
     */
    validateSessions(data) {
        if (!data || typeof data !== 'object') return {};
        
        const validated = {};
        for (const [id, session] of Object.entries(data)) {
            if (this.isValidSession(session)) {
                validated[id] = session;
            }
        }
        return validated;
    },

    /**
     * Validate a single session
     */
    isValidSession(session) {
        return session &&
            typeof session.id === 'string' &&
            typeof session.title === 'string' &&
            typeof session.createdAt === 'number' &&
            typeof session.updatedAt === 'number' &&
            Array.isArray(session.messages);
    },

    /**
     * Validate LLM config
     */
    validateLLMConfig(config) {
        const defaults = {
            provider: '',
            model: '',
            endpoint: '',
            parameters: {}
        };
        
        if (!config || typeof config !== 'object') return defaults;
        
        return {
            provider: typeof config.provider === 'string' ? config.provider : defaults.provider,
            model: typeof config.model === 'string' ? config.model : defaults.model,
            endpoint: typeof config.endpoint === 'string' ? config.endpoint : defaults.endpoint,
            parameters: typeof config.parameters === 'object' ? config.parameters : defaults.parameters
        };
    },

    /**
     * Save sessions atomically to localStorage
     */
    saveSessions() {
        try {
            const json = JSON.stringify(this.state.sessions);
            localStorage.setItem(this.STORAGE_KEYS.SESSIONS, json);
        } catch (e) {
            if (e.name === 'QuotaExceededError') {
                console.error('Storage quota exceeded');
                this.handleStorageQuotaError();
            } else {
                console.error('Failed to save sessions:', e);
            }
        }
    },

    /**
     * Save active session ID
     */
    saveActiveSession() {
        try {
            if (this.state.activeSessionId) {
                localStorage.setItem(this.STORAGE_KEYS.ACTIVE_SESSION, this.state.activeSessionId);
            } else {
                localStorage.removeItem(this.STORAGE_KEYS.ACTIVE_SESSION);
            }
        } catch (e) {
            console.error('Failed to save active session:', e);
        }
    },

    /**
     * Save LLM config
     */
    saveLLMConfig() {
        try {
            localStorage.setItem(this.STORAGE_KEYS.LLM_CONFIG, JSON.stringify(this.state.llmConfig));
        } catch (e) {
            console.error('Failed to save LLM config:', e);
        }
    },

    /**
     * Handle storage quota exceeded
     */
    handleStorageQuotaError() {
        // Remove oldest sessions until we have space
        const sessions = Object.values(this.state.sessions)
            .sort((a, b) => a.updatedAt - b.updatedAt);
        
        while (sessions.length > 1) {
            const oldest = sessions.shift();
            delete this.state.sessions[oldest.id];
            
            try {
                this.saveSessions();
                console.log(`Removed old session ${oldest.id} to free storage`);
                break;
            } catch (e) {
                if (e.name !== 'QuotaExceededError') break;
            }
        }
    },

    /**
     * Handle messages that were streaming when page was closed
     */
    handleInterruptedMessages() {
        for (const session of Object.values(this.state.sessions)) {
            let modified = false;
            for (const message of session.messages) {
                if (message.status === 'streaming') {
                    message.status = 'error';
                    message.errorMessage = 'Interrupted by page reload';
                    modified = true;
                }
            }
            if (modified) {
                session.updatedAt = Date.now();
            }
        }
        if (Object.keys(this.state.sessions).length > 0) {
            this.saveSessions();
        }
    },

    /**
     * Setup beforeunload handler
     */
    setupBeforeUnload() {
        window.addEventListener('beforeunload', () => {
            // Save any pending state
            this.saveSessions();
            this.saveActiveSession();
        });
    },

    // ==================== ACTIONS ====================

    /**
     * Create a new session
     */
    createSession(title = null) {
        const id = this.generateId();
        const now = Date.now();
        
        const session = {
            id,
            title: title || `Chat ${new Date().toLocaleDateString()}`,
            createdAt: now,
            updatedAt: now,
            messages: []
        };
        
        this.state.sessions[id] = session;
        this.state.activeSessionId = id;
        
        this.saveSessions();
        this.saveActiveSession();
        this.notify();
        
        return session;
    },

    /**
     * Select a session as active
     */
    selectSession(sessionId) {
        if (!this.state.sessions[sessionId]) return false;
        
        this.state.activeSessionId = sessionId;
        this.saveActiveSession();
        this.notify();
        
        return true;
    },

    /**
     * Rename a session
     */
    renameSession(sessionId, newTitle) {
        const session = this.state.sessions[sessionId];
        if (!session) return false;
        
        session.title = newTitle;
        session.updatedAt = Date.now();
        
        this.saveSessions();
        this.notify();
        
        return true;
    },

    /**
     * Delete a session
     */
    deleteSession(sessionId) {
        if (!this.state.sessions[sessionId]) return false;
        
        delete this.state.sessions[sessionId];
        
        // If deleted active session, select another
        if (this.state.activeSessionId === sessionId) {
            const remaining = Object.values(this.state.sessions)
                .sort((a, b) => b.updatedAt - a.updatedAt);
            this.state.activeSessionId = remaining[0]?.id || null;
        }
        
        this.saveSessions();
        this.saveActiveSession();
        this.notify();
        
        return true;
    },

    /**
     * Get the active session
     */
    getActiveSession() {
        return this.state.activeSessionId 
            ? this.state.sessions[this.state.activeSessionId] 
            : null;
    },

    /**
     * Add a message to the active session
     */
    addMessage(message) {
        const session = this.getActiveSession();
        if (!session) return null;
        
        const fullMessage = {
            id: message.id || this.generateId(),
            role: message.role,
            content: message.content || '',
            attachments: message.attachments || [],
            status: message.status || 'complete',
            createdAt: message.createdAt || Date.now()
        };
        
        // Check for duplicates
        if (session.messages.some(m => m.id === fullMessage.id)) {
            return null;
        }
        
        session.messages.push(fullMessage);
        session.updatedAt = Date.now();
        
        // Auto-generate title from first user message
        if (session.messages.length === 1 && message.role === 'user') {
            const preview = message.content.slice(0, 50);
            session.title = preview + (message.content.length > 50 ? '...' : '');
        }
        
        this.saveSessions();
        this.notify();
        
        return fullMessage;
    },

    /**
     * Update a message in the active session
     */
    updateMessage(messageId, updates) {
        const session = this.getActiveSession();
        if (!session) return false;
        
        const message = session.messages.find(m => m.id === messageId);
        if (!message) return false;
        
        Object.assign(message, updates);
        session.updatedAt = Date.now();
        
        this.saveSessions();
        this.notify();
        
        return true;
    },

    /**
     * Get all sessions sorted by most recent
     */
    getAllSessions() {
        return Object.values(this.state.sessions)
            .sort((a, b) => b.updatedAt - a.updatedAt);
    },

    /**
     * Set streaming state
     */
    setStreaming(isStreaming) {
        this.state.uiState.isStreaming = isStreaming;
        this.notify();
    },

    /**
     * Set auto-scroll enabled
     */
    setAutoScroll(enabled) {
        this.state.uiState.autoScrollEnabled = enabled;
        this.notify();
    },

    /**
     * Add attachment to draft
     */
    addAttachmentDraft(attachment) {
        this.state.uiState.activeAttachmentDraft.push(attachment);
        this.notify();
    },

    /**
     * Remove attachment from draft
     */
    removeAttachmentDraft(attachmentId) {
        this.state.uiState.activeAttachmentDraft = 
            this.state.uiState.activeAttachmentDraft.filter(a => a.id !== attachmentId);
        this.notify();
    },

    /**
     * Clear attachment draft (after sending)
     */
    clearAttachmentDraft() {
        this.state.uiState.activeAttachmentDraft = [];
        this.notify();
    },

    /**
     * Update LLM config
     */
    updateLLMConfig(config) {
        this.state.llmConfig = this.validateLLMConfig({
            ...this.state.llmConfig,
            ...config
        });
        this.saveLLMConfig();
        this.notify();
    },

    /**
     * Export LLM config as JSON
     */
    exportLLMConfig() {
        return JSON.stringify(this.state.llmConfig, null, 2);
    },

    /**
     * Import LLM config from JSON
     */
    importLLMConfig(jsonString) {
        try {
            const parsed = JSON.parse(jsonString);
            const validated = this.validateLLMConfig(parsed);
            
            // Check if config is valid (has at least provider and model)
            if (!validated.provider && !validated.model) {
                return { success: false, error: 'Invalid config: missing provider and model' };
            }
            
            this.state.llmConfig = validated;
            this.saveLLMConfig();
            this.notify();
            
            return { success: true };
        } catch (e) {
            return { success: false, error: 'Invalid JSON format' };
        }
    },

    /**
     * Export all sessions
     */
    exportSessions() {
        return JSON.stringify({
            sessions: this.state.sessions,
            exportedAt: Date.now()
        }, null, 2);
    },

    /**
     * Import sessions
     */
    importSessions(jsonString) {
        try {
            const parsed = JSON.parse(jsonString);
            const sessions = parsed.sessions || parsed;
            
            const validated = this.validateSessions(sessions);
            const count = Object.keys(validated).length;
            
            if (count === 0) {
                return { success: false, error: 'No valid sessions found' };
            }
            
            // Merge with existing sessions
            this.state.sessions = { ...this.state.sessions, ...validated };
            this.saveSessions();
            this.notify();
            
            return { success: true, count };
        } catch (e) {
            return { success: false, error: 'Invalid JSON format' };
        }
    },

    /**
     * Generate unique ID
     */
    generateId() {
        return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
};

// Export
window.Store = Store;
