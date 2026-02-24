/**
 * Settings Panel Controller
 * Handles LLM credential management, import/export, and status display
 */

const Settings = {
    isOpen: false,
    THEME_KEY: 'zededa_theme',

    /**
     * Initialize settings panel
     */
    init() {
        // Event listeners
        this.bindEvents();

        // Initialize theme
        this.initTheme();
        
        // Initialize provider type UI
        this.onProviderTypeChange();
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Open/close buttons
        document.getElementById('settings-btn')?.addEventListener('click', () => this.open());
        document.getElementById('settings-close-btn')?.addEventListener('click', () => this.close());
        
        // Overlay click to close
        document.getElementById('settings-overlay')?.addEventListener('click', (e) => {
            if (e.target.id === 'settings-overlay') this.close();
        });

        // Escape key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) this.close();
        });

        // Provider type change
        document.getElementById('cred-provider-type')?.addEventListener('change', () => this.onProviderTypeChange());

        // Save and activate button
        document.getElementById('save-activate-btn')?.addEventListener('click', () => this.saveAndActivate());

        // Fetch models button
        document.getElementById('fetch-models-btn')?.addEventListener('click', () => this.fetchModels());

        // Import/Export
        document.getElementById('export-config-btn')?.addEventListener('click', () => this.exportConfig());
        document.getElementById('import-config-btn')?.addEventListener('click', () => {
            document.getElementById('import-file-input')?.click();
        });
        document.getElementById('import-file-input')?.addEventListener('change', (e) => this.importConfig(e));

        // Theme toggle
        document.getElementById('theme-toggle-btn')?.addEventListener('click', () => this.toggleTheme());
    },

    /**
     * Open settings panel
     */
    open() {
        const overlay = document.getElementById('settings-overlay');
        if (overlay) {
            overlay.classList.remove('hidden');
            this.isOpen = true;
            this.updateThemeButton();
            this.loadCredentials();
            this.loadRouterStatus();
            this.loadServerStatus();
        }
    },

    /**
     * Initialize theme from localStorage
     */
    initTheme() {
        const savedTheme = localStorage.getItem(this.THEME_KEY) || 'light';
        this.applyTheme(savedTheme, false);
    },

    /**
     * Toggle between light and dark theme
     */
    toggleTheme() {
        const currentTheme = this.getCurrentTheme();
        const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(nextTheme, true);
    },

    /**
     * Apply selected theme
     */
    applyTheme(theme, showToast = false) {
        const normalizedTheme = theme === 'dark' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', normalizedTheme);
        localStorage.setItem(this.THEME_KEY, normalizedTheme);
        this.updateThemeAssets(normalizedTheme);
        this.updateThemeButton();

        if (showToast) {
            Utils.showToast(`${normalizedTheme === 'dark' ? 'Dark' : 'Light'} theme enabled`, 'info');
        }
    },

    /**
     * Get current active theme
     */
    getCurrentTheme() {
        return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    },

    /**
     * Update theme toggle button label/icon
     */
    updateThemeButton() {
        const btn = document.getElementById('theme-toggle-btn');
        const label = document.getElementById('theme-toggle-label');
        if (!btn || !label) return;

        const isDark = this.getCurrentTheme() === 'dark';
        label.textContent = isDark ? 'Enable Light Theme' : 'Enable Dark Theme';

        const icon = btn.querySelector('i');
        if (icon) {
            icon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
        }
    },

    /**
     * Update theme-dependent visual assets
     */
    updateThemeAssets(theme) {
        const headerLogo = document.getElementById('header-logo');
        if (headerLogo) {
            headerLogo.src = theme === 'dark' ? '/static/logo-dark.png' : '/static/logo-light.png';
        }
    },

    /**
     * Close settings panel
     */
    close() {
        const overlay = document.getElementById('settings-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
            this.isOpen = false;
        }
    },

    /**
     * Handle provider type change
     */
    onProviderTypeChange() {
        const providerType = document.getElementById('cred-provider-type')?.value || 'lmstudio';
        const isLocal = ['lmstudio', 'ollama', 'vllm', 'tgi', 'openai-compatible'].includes(providerType);
        const isCloud = ['openai', 'anthropic', 'google', 'groq'].includes(providerType);

        // Update API key hint
        const apiKeyHint = document.getElementById('api-key-hint');
        if (apiKeyHint) {
            apiKeyHint.textContent = isLocal ? '(optional for local servers)' : '(required)';
            apiKeyHint.className = isLocal ? 'form-hint' : 'form-hint text-error';
        }

        // Update URL hint
        const urlHint = document.getElementById('url-hint');
        if (urlHint) {
            urlHint.textContent = isCloud ? '(optional)' : '(required)';
            urlHint.className = isCloud ? 'form-hint' : 'form-hint text-error';
        }

        // Update placeholders
        const urlInput = document.getElementById('cred-url');
        const modelInput = document.getElementById('cred-model');

        const placeholders = {
            lmstudio: { url: 'http://192.168.x.x:1234/v1', model: 'e.g., google/gemma-3-4b' },
            ollama: { url: 'http://localhost:11434', model: 'e.g., llama3.2' },
            vllm: { url: 'http://localhost:8000/v1', model: 'model name' },
            tgi: { url: 'http://localhost:8080', model: 'model name' },
            'openai-compatible': { url: 'http://localhost:8000/v1', model: 'model name' },
            openai: { url: 'https://api.openai.com/v1 (default)', model: 'gpt-4o' },
            anthropic: { url: '(not needed)', model: 'claude-sonnet-4-20250514' },
            google: { url: '(not needed)', model: 'gemini-1.5-pro' },
            groq: { url: '(not needed)', model: 'Click Fetch to load models' }
        };

        const config = placeholders[providerType] || placeholders['openai-compatible'];
        if (urlInput) urlInput.placeholder = config.url;
        if (modelInput) modelInput.placeholder = config.model;

        // Reset model dropdown
        this.resetModelDropdown();
    },

    /**
     * Reset model dropdown to text input mode
     */
    resetModelDropdown() {
        const modelSelect = document.getElementById('cred-model-select');
        const modelInput = document.getElementById('cred-model');
        const fetchStatus = document.getElementById('fetch-status');

        if (modelSelect) modelSelect.classList.add('hidden');
        if (modelInput) {
            modelInput.classList.remove('hidden');
            modelInput.value = '';
        }
        if (fetchStatus) fetchStatus.textContent = '';
    },

    /**
     * Fetch available models from provider
     */
    async fetchModels() {
        const providerType = document.getElementById('cred-provider-type')?.value;
        const apiKey = document.getElementById('cred-api-key')?.value.trim();
        const url = document.getElementById('cred-url')?.value.trim();
        const fetchBtn = document.getElementById('fetch-models-btn');
        const fetchStatus = document.getElementById('fetch-status');
        const modelSelect = document.getElementById('cred-model-select');
        const modelInput = document.getElementById('cred-model');

        const isCloud = ['openai', 'anthropic', 'google', 'groq'].includes(providerType);
        const isLocal = !isCloud;

        // Validate
        if (isCloud && !apiKey) {
            if (fetchStatus) {
                fetchStatus.textContent = '⚠️ API key required';
                fetchStatus.className = 'form-hint text-error';
            }
            return;
        }

        if (isLocal && !url) {
            if (fetchStatus) {
                fetchStatus.textContent = '⚠️ Server URL required';
                fetchStatus.className = 'form-hint text-error';
            }
            return;
        }

        // Loading state
        if (fetchBtn) {
            fetchBtn.disabled = true;
            fetchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        }
        if (fetchStatus) {
            fetchStatus.textContent = 'Fetching models...';
            fetchStatus.className = 'form-hint';
        }

        try {
            const response = await fetch('/llm/models/fetch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider_type: providerType,
                    api_key: apiKey || null,
                    url: url || null
                })
            });

            const data = await response.json();

            if (data.success && data.models?.length > 0) {
                // Populate dropdown
                if (modelSelect) {
                    modelSelect.innerHTML = '<option value="">-- Select a model --</option>';
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        modelSelect.appendChild(option);
                    });
                    modelSelect.classList.remove('hidden');
                }
                if (modelInput) modelInput.classList.add('hidden');
                if (fetchStatus) {
                    fetchStatus.textContent = `✓ Found ${data.models.length} models`;
                    fetchStatus.className = 'form-hint text-success';
                }
            } else {
                if (fetchStatus) {
                    fetchStatus.textContent = '⚠️ No models found. Enter manually.';
                    fetchStatus.className = 'form-hint text-warning';
                }
            }
        } catch (err) {
            console.error('Failed to fetch models:', err);
            if (fetchStatus) {
                fetchStatus.textContent = `✗ ${err.message}`;
                fetchStatus.className = 'form-hint text-error';
            }
        } finally {
            if (fetchBtn) {
                fetchBtn.disabled = false;
                fetchBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Fetch';
            }
        }
    },

    /**
     * Save and activate credential
     */
    async saveAndActivate() {
        const name = document.getElementById('cred-name')?.value.trim();
        const providerType = document.getElementById('cred-provider-type')?.value;
        const apiKey = document.getElementById('cred-api-key')?.value.trim();
        const url = document.getElementById('cred-url')?.value.trim();
        const supportsTools = document.getElementById('cred-supports-tools')?.checked ?? true;

        // Get model from dropdown if visible, otherwise from input
        const modelSelect = document.getElementById('cred-model-select');
        const modelInput = document.getElementById('cred-model');
        const model = (modelSelect && !modelSelect.classList.contains('hidden'))
            ? modelSelect.value
            : modelInput?.value.trim();

        // Validation
        if (!name) {
            alert('Please enter a configuration name');
            return;
        }

        const isLocal = ['lmstudio', 'ollama', 'vllm', 'tgi', 'openai-compatible'].includes(providerType);
        if (isLocal && !url) {
            alert('Please enter the server URL');
            return;
        }

        if (!model) {
            alert('Please select or enter a model name');
            return;
        }

        const btn = document.getElementById('save-activate-btn');
        const originalHTML = btn?.innerHTML;
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
        }

        try {
            const payload = {
                name,
                provider_type: providerType,
                supports_tools: supportsTools,
                priority: 1
            };
            if (apiKey) payload.api_key = apiKey;
            if (url) payload.url = url;
            if (model) payload.model = model;

            const response = await fetch('/llm/credentials', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (data.success) {
                // Clear form
                document.getElementById('cred-name').value = '';
                document.getElementById('cred-api-key').value = '';
                document.getElementById('cred-url').value = '';
                document.getElementById('cred-model').value = '';
                this.resetModelDropdown();

                // Reload list and activate
                await this.loadCredentials();
                await this.activateCredential(name);
            } else {
                alert('Failed to save: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error('Failed to save:', err);
            alert('Failed to save: ' + err.message);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = originalHTML;
            }
        }
    },

    /**
     * Load saved credentials
     */
    async loadCredentials() {
        const container = document.getElementById('credentials-list');
        if (!container) return;

        container.innerHTML = '<div class="loading-state"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';

        try {
            const response = await fetch('/llm/credentials');
            const data = await response.json();

            if (data.success && data.credentials?.length > 0) {
                container.innerHTML = '';
                
                data.credentials.forEach(cred => {
                    const item = Utils.createElement('div', {
                        className: `credential-item ${cred.enabled ? 'active' : ''}`
                    });

                    item.innerHTML = `
                        <div class="credential-info">
                            <div class="credential-name">${Utils.escapeHtml(cred.name)}</div>
                            <div class="credential-meta">
                                ${Utils.escapeHtml(cred.provider_type)}
                                ${cred.model ? ' • ' + Utils.escapeHtml(cred.model) : ''}
                                ${cred.has_api_key ? ' • <i class="fas fa-key" style="color: var(--color-success);"></i>' : ''}
                            </div>
                        </div>
                        <div class="credential-actions">
                            <button class="btn btn-icon btn-ghost" onclick="Settings.activateCredential('${Utils.escapeHtml(cred.name)}')" title="Activate">
                                <i class="fas fa-play"></i>
                            </button>
                            <button class="btn btn-icon btn-ghost" onclick="Settings.deleteCredential('${Utils.escapeHtml(cred.name)}')" title="Delete">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    `;

                    container.appendChild(item);
                });
            } else {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon"><i class="fas fa-inbox"></i></div>
                        <div class="empty-state-description">No configurations saved yet</div>
                    </div>
                `;
            }
        } catch (err) {
            console.error('Failed to load credentials:', err);
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon"><i class="fas fa-exclamation-circle"></i></div>
                    <div class="empty-state-description">Failed to load credentials</div>
                </div>
            `;
        }
    },

    /**
     * Activate a credential
     */
    async activateCredential(name) {
        try {
            const response = await fetch(`/llm/credentials/${encodeURIComponent(name)}/activate`, {
                method: 'POST'
            });
            const data = await response.json();

            if (data.success && data.activated) {
                // Refresh status displays
                await this.loadRouterStatus();
                await this.loadCredentials();

                // Check if actually connected
                const statusResp = await fetch('/llm/status');
                const statusData = await statusResp.json();

                if (statusData.success) {
                    const provider = (statusData.providers || []).find(p => p.name === name);
                    if (provider?.status?.available) {
                        Utils.showToast(`${name} activated successfully`, 'success');
                    } else {
                        const errorMsg = provider?.status?.last_error || 'Could not connect';
                        Utils.showToast(`${name} registered but unreachable: ${errorMsg}`, 'warning');
                    }
                }

                // Start a fresh chat session when provider changes
                if (window.Chat) {
                    Chat.clearChat();
                    Chat.showWelcome();
                    Chat.checkStatus();
                }
                if (window.Sidebar) {
                    Sidebar.createNewSession();
                }
            } else {
                alert('Failed to activate: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error('Failed to activate:', err);
            alert('Failed to activate: ' + err.message);
        }
    },

    /**
     * Delete a credential
     */
    async deleteCredential(name) {
        if (!confirm(`Delete "${name}"? This cannot be undone.`)) return;

        try {
            const response = await fetch(`/llm/credentials/${encodeURIComponent(name)}`, {
                method: 'DELETE'
            });
            const data = await response.json();

            if (data.success) {
                await this.loadCredentials();
                await this.loadRouterStatus();
                Utils.showToast('Credential deleted', 'info');
            } else {
                alert('Failed to delete: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error('Failed to delete:', err);
            alert('Failed to delete: ' + err.message);
        }
    },

    /**
     * Export configuration
     */
    async exportConfig() {
        try {
            const response = await fetch('/llm/credentials/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const data = await response.json();

            if (data.success && data.bundle) {
                Utils.downloadFile(
                    JSON.stringify(data.bundle, null, 2),
                    `llm-config-${Date.now()}.json`,
                    'application/json'
                );
                Utils.showToast('Configuration exported', 'success');
            } else {
                alert('Failed to export: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error('Failed to export:', err);
            alert('Failed to export: ' + err.message);
        }
    },

    /**
     * Import configuration
     */
    async importConfig(event) {
        const file = event.target.files?.[0];
        if (!file) return;

        try {
            const text = await Utils.readFileAsText(file);
            const bundle = JSON.parse(text);

            const response = await fetch('/llm/credentials/import', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ bundle, overwrite: false })
            });
            const data = await response.json();

            if (data.success) {
                const imported = data.imported_count || 0;
                const skipped = data.skipped_count || 0;

                if (imported > 0) {
                    // Auto-activate all
                    const activateResp = await fetch('/llm/credentials/activate-all', {
                        method: 'POST'
                    });
                    const activateData = await activateResp.json();

                    Utils.showToast(`Imported ${imported} configuration(s)`, 'success');
                } else if (skipped > 0) {
                    Utils.showToast(`${skipped} configuration(s) already exist`, 'info');
                }

                await this.loadCredentials();
                await this.loadRouterStatus();

                // Start a fresh chat session after import
                if (window.Chat) {
                    Chat.clearChat();
                    Chat.showWelcome();
                    Chat.checkStatus();
                }
                if (window.Sidebar) {
                    Sidebar.createNewSession();
                }
            } else {
                alert('Failed to import: ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            console.error('Failed to import:', err);
            alert('Failed to import: Invalid JSON file');
        }

        // Clear file input
        event.target.value = '';
    },

    /**
     * Load LLM router status
     */
    async loadRouterStatus() {
        const container = document.getElementById('router-status');
        if (!container) return;

        try {
            const response = await fetch('/llm/status');
            const data = await response.json();

            if (data.success) {
                const providers = data.providers || [];
                const active = data.active_provider;

                let html = '<div class="status-card">';
                html += `<div class="status-row">
                    <span class="status-label">Active Provider</span>
                    <span class="status-value ${active ? 'success' : 'error'}">${active ? Utils.escapeHtml(active.name) : 'None'}</span>
                </div>`;
                html += `<div class="status-row">
                    <span class="status-label">Registered</span>
                    <span class="status-value">${providers.length}</span>
                </div>`;

                if (providers.length > 0) {
                    providers.forEach(p => {
                        const isAvailable = p.status?.available;
                        const badge = isAvailable
                            ? '<span class="badge badge-success">Connected</span>'
                            : '<span class="badge badge-error">Offline</span>';
                        html += `<div class="status-row">
                            <span class="status-label">${Utils.escapeHtml(p.name)}</span>
                            ${badge}
                        </div>`;
                    });
                }

                html += '</div>';
                container.innerHTML = html;

                // Update header status indicator
                this.updateHeaderStatus(active, providers);
            } else {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-description">Failed to load status</div></div>';
            }
        } catch (err) {
            console.error('Failed to load router status:', err);
            container.innerHTML = '<div class="empty-state"><div class="empty-state-description">Router not available</div></div>';
        }
    },

    /**
     * Load inference server status
     */
    async loadServerStatus() {
        const container = document.getElementById('server-status');
        if (!container) return;

        try {
            const response = await fetch('/health');
            const data = await response.json();

            let html = '<div class="status-card">';
            html += `<div class="status-row">
                <span class="status-label">Status</span>
                <span class="status-value ${data.success ? 'success' : 'error'}">${data.status || (data.success ? 'Healthy' : 'Unhealthy')}</span>
            </div>`;
            html += `<div class="status-row">
                <span class="status-label">Server Type</span>
                <span class="status-value">${Utils.escapeHtml(data.server_type || 'Unknown')}</span>
            </div>`;
            html += `<div class="status-row">
                <span class="status-label">Models</span>
                <span class="status-value">${(data.available_models || []).length}</span>
            </div>`;
            html += '</div>';

            container.innerHTML = html;
        } catch (err) {
            console.error('Failed to load server status:', err);
            container.innerHTML = '<div class="empty-state"><div class="empty-state-description">Server not available</div></div>';
        }
    },

    /**
     * Update header status indicator
     */
    updateHeaderStatus(active, providers) {
        const statusDot = document.getElementById('header-status-dot');
        const statusText = document.getElementById('header-status-text');

        if (!statusDot || !statusText) return;

        if (active) {
            statusDot.className = 'status-dot active';
            statusText.textContent = active.name;
        } else if (providers?.length > 0) {
            statusDot.className = 'status-dot warning';
            statusText.textContent = 'Disconnected';
        } else {
            statusDot.className = 'status-dot';
            statusText.textContent = 'Not configured';
        }
    }
};

// Export
window.Settings = Settings;
