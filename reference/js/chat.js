/**
 * Chat Controller
 * Handles chat interface, messages, attachments, and streaming agent communication
 * Integrates with Store for session persistence
 */

const Chat = {
    sessionId: null,
    enabled: false,
    attachments: [],
    isLoading: false,
    streamingMessage: null,  // Currently streaming message element
    streamingMessageId: null, // ID of the message being streamed
    streamingContent: '',    // Accumulated streamed content
    pendingTokens: '',       // Tokens waiting to be rendered
    isRendering: false,      // Prevents overlapping render cycles
    lastRenderTime: 0,       // Last render timestamp for throttling
    autoScrollEnabled: true, // Auto-scroll to bottom behavior
    userScrolledUp: false,   // Track if user has scrolled up

    /**
     * Initialize chat
     */
    init() {
        this.bindEvents();
        this.setupScrollTracking();
        this.checkStatus();
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Send button
        document.getElementById('chat-send-btn')?.addEventListener('click', () => this.send());

        // Textarea - Enter to send, Shift+Enter for newline
        const textarea = document.getElementById('chat-input');
        textarea?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.send();
            }
        });

        // Auto-resize textarea
        textarea?.addEventListener('input', () => this.resizeTextarea());

        // Attachment button
        document.getElementById('attachment-btn')?.addEventListener('click', () => this.toggleAttachmentMenu());

        // Attachment menu items
        document.getElementById('attach-image-btn')?.addEventListener('click', () => {
            document.getElementById('attach-image-input')?.click();
            this.hideAttachmentMenu();
        });
        document.getElementById('attach-file-btn')?.addEventListener('click', () => {
            document.getElementById('attach-file-input')?.click();
            this.hideAttachmentMenu();
        });

        // File inputs
        document.getElementById('attach-image-input')?.addEventListener('change', (e) => this.handleFileSelect(e, 'image'));
        document.getElementById('attach-file-input')?.addEventListener('change', (e) => this.handleFileSelect(e, 'file'));

        // Click outside to close attachment menu
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.attachment-btn') && !e.target.closest('.attachment-menu')) {
                this.hideAttachmentMenu();
            }
        });

        // Image modal close
        document.getElementById('image-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'image-modal' || e.target.closest('.image-modal-close')) {
                this.closeImageModal();
            }
        });
    },

    /**
     * Setup scroll tracking for auto-scroll behavior
     */
    setupScrollTracking() {
        const messagesDiv = document.getElementById('chat-messages');
        if (!messagesDiv) return;

        messagesDiv.addEventListener('scroll', () => {
            const { scrollTop, scrollHeight, clientHeight } = messagesDiv;
            const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
            
            // If user scrolled more than 100px from bottom, disable auto-scroll
            if (distanceFromBottom > 100) {
                this.userScrolledUp = true;
                this.autoScrollEnabled = false;
            } else {
                // User scrolled back to bottom, re-enable auto-scroll
                this.userScrolledUp = false;
                this.autoScrollEnabled = true;
            }
        });
    },

    /**
     * Load a session's messages into the chat
     */
    loadSession(sessionId) {
        const session = Store.state.sessions[sessionId];
        if (!session) return;

        this.sessionId = sessionId;
        this.clearChat();

        if (session.messages.length === 0) {
            this.showWelcome();
            return;
        }

        const inner = this.ensureMessagesInner();
        if (!inner) return;

        // Render all messages
        for (const message of session.messages) {
            this.renderStoredMessage(message, inner);
        }

        this.scrollToBottom(true); // Force scroll to bottom on load
    },

    /**
     * Render a message from storage
     */
    renderStoredMessage(message, container) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${message.role}`;
        messageDiv.dataset.messageId = message.id;

        // Build content based on role
        let htmlContent = '';
        if (message.role === 'user') {
            htmlContent = Utils.escapeHtml(message.content).replace(/\n/g, '<br>');
        } else if (message.role === 'agent') {
            htmlContent = Markdown.render(message.content || '');
        } else {
            htmlContent = Utils.escapeHtml(message.content);
        }

        // Check for error status
        const isError = message.status === 'error';
        const errorMessage = message.errorMessage || 'An error occurred';

        messageDiv.innerHTML = `
            <div class="chat-content">
                <div class="chat-bubble ${isError ? 'chat-bubble-error' : ''}">
                    ${message.attachments?.length > 0 && message.role === 'user' ? '<div class="chat-attached-images"></div>' : ''}
                    <div class="chat-text">${htmlContent}</div>
                    ${isError ? `
                        <div class="chat-error-info">
                            <span class="chat-error-message"><i class="fas fa-exclamation-circle"></i> ${Utils.escapeHtml(errorMessage)}</span>
                            ${message.role === 'agent' ? `<button class="chat-retry-btn" onclick="Chat.retryMessage('${message.id}')"><i class="fas fa-redo"></i> Retry</button>` : ''}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;

        // Add attachment thumbnails for user messages
        if (message.attachments?.length > 0 && message.role === 'user') {
            const attachedContainer = messageDiv.querySelector('.chat-attached-images');
            if (attachedContainer) {
                message.attachments.forEach(att => {
                    if (att.type === 'image' && att.previewUrl) {
                        const imgWrapper = document.createElement('div');
                        imgWrapper.className = 'chat-attached-image';
                        imgWrapper.innerHTML = `
                            <img src="${att.previewUrl}" alt="${Utils.escapeHtml(att.name)}" onclick="Chat.showImage(this.src)">
                            <span class="chat-attached-filename">${Utils.escapeHtml(att.name)}</span>
                        `;
                        attachedContainer.appendChild(imgWrapper);
                    } else {
                        const fileWrapper = document.createElement('div');
                        fileWrapper.className = 'chat-attached-file';
                        const icon = att.name?.endsWith('.pdf') ? 'fa-file-pdf' : 'fa-file-alt';
                        fileWrapper.innerHTML = `
                            <i class="fas ${icon}"></i>
                            <span>${Utils.escapeHtml(att.name)}</span>
                        `;
                        attachedContainer.appendChild(fileWrapper);
                    }
                });
            }
        }

        container.appendChild(messageDiv);
    },

    /**
     * Retry a failed message
     */
    async retryMessage(messageId) {
        const session = Store.getActiveSession();
        if (!session) return;

        // Find the failed agent message
        const messageIndex = session.messages.findIndex(m => m.id === messageId);
        if (messageIndex === -1) return;

        // Find the preceding user message
        let userMessage = null;
        for (let i = messageIndex - 1; i >= 0; i--) {
            if (session.messages[i].role === 'user') {
                userMessage = session.messages[i];
                break;
            }
        }

        if (!userMessage) return;

        // Remove the failed message from store
        session.messages.splice(messageIndex, 1);
        session.updatedAt = Date.now();
        Store.saveSessions();

        // Remove from UI
        const messageEl = document.querySelector(`[data-message-id="${messageId}"]`);
        if (messageEl) messageEl.remove();

        // Resend the user message
        this.showTyping();
        this.setLoading(true);
        await this.sendStreaming(userMessage.content);
    },

    /**
     * Clear the chat display
     */
    clearChat() {
        const messagesDiv = document.getElementById('chat-messages');
        if (messagesDiv) {
            messagesDiv.innerHTML = '<div class="chat-messages-inner"></div>';
        }
        this.autoScrollEnabled = true;
        this.userScrolledUp = false;
    },

    /**
     * Check if agent is enabled
     */
    async checkStatus() {
        try {
            const response = await fetch('/agent/status');
            const data = await response.json();

            this.enabled = data.enabled;
            
            // Update header status indicator
            const llmRouter = data.llm_router || {};
            this.updateHeaderStatus(llmRouter);

            if (!this.enabled) {
                this.showConfigRequired();
            } else {
                const hasActiveProvider = !!llmRouter.active_provider;

                if (!hasActiveProvider && llmRouter.providers > 0) {
                    this.showConnectionPending(data.message || 'LLM server is not reachable.');
                } else if (!hasActiveProvider) {
                    this.showConfigRequired();
                } else {
                    // Load active session or show welcome
                    this.enableInput();
                    this.loadActiveSession();
                }
            }
        } catch (err) {
            console.error('Failed to check agent status:', err);
            this.showConfigRequired('Failed to connect to agent service');
            this.updateHeaderStatus(null);
        }
    },

    /**
     * Load active session from Store or create new one
     */
    loadActiveSession() {
        const activeSession = Store.getActiveSession();
        
        if (activeSession && activeSession.messages.length > 0) {
            this.loadSession(activeSession.id);
        } else if (activeSession) {
            this.sessionId = activeSession.id;
            this.showWelcome();
        } else {
            // Create a new session
            const session = Store.createSession();
            this.sessionId = session.id;
            this.showWelcome();
        }
    },
    
    /**
     * Update header status indicator based on LLM router state
     */
    updateHeaderStatus(llmRouter) {
        const statusDot = document.getElementById('header-status-dot');
        const statusText = document.getElementById('header-status-text');
        
        if (!statusDot || !statusText) return;
        
        if (llmRouter?.active_provider) {
            statusDot.className = 'status-dot active';
            // Handle both string and object formats
            const providerName = typeof llmRouter.active_provider === 'string' 
                ? llmRouter.active_provider 
                : llmRouter.active_provider.name;
            statusText.textContent = providerName || 'Connected';
        } else if (llmRouter?.providers > 0) {
            statusDot.className = 'status-dot warning';
            statusText.textContent = 'Disconnected';
        } else {
            statusDot.className = 'status-dot';
            statusText.textContent = 'Not configured';
        }
    },

    /**
     * Show configuration required state
     */
    showConfigRequired(message) {
        const messagesDiv = document.getElementById('chat-messages');
        if (!messagesDiv) return;

        this.disableInput();

        messagesDiv.innerHTML = `
            <div class="chat-messages-inner">
                <div class="chat-config-required">
                    <div class="chat-config-required-icon">⚙️</div>
                    <h2>Configuration Required</h2>
                    <p>${message || 'Configure an LLM provider to enable the AI assistant.'}</p>
                    <button class="config-btn" onclick="Settings.open()">
                        <i class="fas fa-cog"></i> Configure LLM Provider
                    </button>
                </div>
            </div>
        `;
    },

    /**
     * Show connection pending state
     */
    showConnectionPending(message) {
        const messagesDiv = document.getElementById('chat-messages');
        if (!messagesDiv) return;

        messagesDiv.innerHTML = `
            <div class="chat-messages-inner">
                <div class="chat-config-required">
                    <div class="chat-config-required-icon">⚠️</div>
                    <h2>Connection Failed</h2>
                    <p>${Utils.escapeHtml(message)}</p>
                    <button class="config-btn" onclick="Settings.open()">
                        <i class="fas fa-cog"></i> Check Settings
                    </button>
                </div>
            </div>
        `;
    },

    /**
     * Show welcome message
     */
    showWelcome() {
        const messagesDiv = document.getElementById('chat-messages');
        if (!messagesDiv) return;

        messagesDiv.innerHTML = `
            <div class="chat-messages-inner">
                <div class="chat-welcome">
                    <div class="chat-welcome-icon">👋</div>
                    <h2>Welcome to ZEDEDA OnDevice Assistant</h2>
                    <h3>Glad to meet you on the device!</h3>
                    <p>I can help you discover deployed ML models, understand their inputs and outputs, run inference, and generate integration code.</p>
                    <div class="chat-suggestions">
                        <button class="suggestion-btn" onclick="Chat.sendSuggestion('What models are available?')">
                            <i class="fas fa-search"></i> What models are available?
                        </button>
                        <button class="suggestion-btn" onclick="Chat.sendSuggestion('Check the server status')">
                            <i class="fas fa-server"></i> Check server status
                        </button>
                        <button class="suggestion-btn" onclick="Chat.sendSuggestion('What should I do next?')">
                            <i class="fas fa-compass"></i> What should I do next?
                        </button>
                        <button class="suggestion-btn" onclick="Chat.sendSuggestion('Help me integrate a model')">
                            <i class="fas fa-code"></i> Help me integrate
                        </button>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Send a suggestion
     */
    sendSuggestion(text) {
        const textarea = document.getElementById('chat-input');
        if (textarea) {
            textarea.value = text;
            this.send();
        }
    },

    /**
     * Ensure messages inner container exists
     */
    ensureMessagesInner() {
        const messagesDiv = document.getElementById('chat-messages');
        if (!messagesDiv) return null;
        
        let inner = messagesDiv.querySelector('.chat-messages-inner');
        if (!inner) {
            messagesDiv.innerHTML = '<div class="chat-messages-inner"></div>';
            inner = messagesDiv.querySelector('.chat-messages-inner');
        }
        
        // Clear welcome/config messages
        const welcome = inner.querySelector('.chat-welcome, .chat-config-required');
        if (welcome) welcome.remove();
        
        return inner;
    },

    /**
     * Add a message to the chat
     * @param {string} role - 'user', 'assistant', or 'system'
     * @param {string} content - Text content of the message
     * @param {Array} images - Result images from inference
     * @param {Array} attachedImages - User-attached image previews
     */
    addMessage(role, content, images = [], attachedImages = []) {
        const inner = this.ensureMessagesInner();
        if (!inner) return null;

        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;

        // Render content
        let htmlContent = role === 'user' 
            ? Utils.escapeHtml(content).replace(/\n/g, '<br>') 
            : Markdown.render(content);

        // Build message HTML - simplified without avatars for cleaner look
        messageDiv.innerHTML = `
            <div class="chat-content">
                <div class="chat-bubble">
                    ${role === 'user' && attachedImages.length > 0 ? '<div class="chat-attached-images"></div>' : ''}
                    <div class="chat-text">${htmlContent}</div>
                </div>
            </div>
        `;

        // Add attached image thumbnails for user messages
        if (role === 'user' && attachedImages.length > 0) {
            const attachedContainer = messageDiv.querySelector('.chat-attached-images');
            if (attachedContainer) {
                attachedImages.forEach(img => {
                    const imgWrapper = document.createElement('div');
                    imgWrapper.className = 'chat-attached-image';
                    imgWrapper.innerHTML = `
                        <img src="${img.preview}" alt="${Utils.escapeHtml(img.name)}" onclick="Chat.showImage(this.src)">
                        <span class="chat-attached-filename">${Utils.escapeHtml(img.name)}</span>
                    `;
                    attachedContainer.appendChild(imgWrapper);
                });
            }
        }

        // Add images if present
        if (images.length > 0) {
            const resultsDiv = document.createElement('div');
            resultsDiv.className = 'chat-results';

            images.forEach(img => {
                const card = document.createElement('div');
                card.className = 'chat-result-card';
                card.innerHTML = `
                    <img src="data:image/png;base64,${img.image}" 
                         alt="${Utils.escapeHtml(img.summary || 'Result')}" 
                         onclick="Chat.showImage(this.src)">
                    <div class="chat-result-caption">
                        <span>${Utils.escapeHtml(img.summary || 'Result')}</span>
                        <button class="btn btn-sm btn-ghost" onclick="Utils.downloadBase64Image('${img.image}', 'result.png')">
                            <i class="fas fa-download"></i>
                        </button>
                    </div>
                `;
                resultsDiv.appendChild(card);
            });

            const contentDiv = messageDiv.querySelector('.chat-content');
            contentDiv?.appendChild(resultsDiv);
        }

        inner.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    },

    /**
     * Create a streaming message placeholder
     * Note: Cursor is only shown once real tokens start arriving
     */
    createStreamingMessage() {
        const inner = this.ensureMessagesInner();
        if (!inner) return null;

        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message assistant';
        messageDiv.id = 'streaming-message';
        
        // Initially empty - cursor will be added when real tokens arrive
        messageDiv.innerHTML = `
            <div class="chat-content">
                <div class="chat-bubble">
                    <div class="chat-text"></div>
                </div>
            </div>
        `;

        inner.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    },

    /**
     * Update streaming message with new content - immediate, no throttling
     * Cursor is shown only during real streaming (when isRealStreaming is true)
     */
    updateStreamingMessage(content) {
        if (!this.streamingMessage) return;
        
        const textDiv = this.streamingMessage.querySelector('.chat-text');
        if (!textDiv) return;
        
        // Render markdown with cursor only if we're receiving real streaming tokens
        const rendered = Markdown.render(content);
        if (this.isRealStreaming) {
            textDiv.innerHTML = rendered + '<span class="streaming-cursor"></span>';
        } else {
            textDiv.innerHTML = rendered;
        }
        
        this.scrollToBottom();
    },
    
    /**
     * Schedule incremental render of accumulated tokens
     * Uses requestAnimationFrame for smooth progressive rendering
     */
    scheduleIncrementalRender() {
        if (this.isRendering || !this.streamingMessage) return;
        
        const now = performance.now();
        const timeSinceLastRender = now - this.lastRenderTime;
        const minRenderInterval = 30; // ~33fps for smooth text appearance
        
        if (timeSinceLastRender < minRenderInterval) {
            // Schedule for later
            setTimeout(() => this.scheduleIncrementalRender(), minRenderInterval - timeSinceLastRender);
            return;
        }
        
        this.isRendering = true;
        this.lastRenderTime = now;
        
        requestAnimationFrame(() => {
            if (this.pendingTokens && this.streamingMessage) {
                this.streamingContent += this.pendingTokens;
                this.pendingTokens = '';
                this.updateStreamingMessage(this.streamingContent);
            }
            this.isRendering = false;
            
            // If more tokens arrived during render, schedule another
            if (this.pendingTokens) {
                this.scheduleIncrementalRender();
            }
        });
    },
    
    /**
     * Append token to pending buffer and trigger render
     */
    appendToken(token) {
        this.pendingTokens += token;
        this.scheduleIncrementalRender();
    },

    /**
     * Finalize streaming message (remove cursor, final render)
     */
    finalizeStreamingMessage(finalContent, images = []) {
        if (!this.streamingMessage) return;
        
        const textDiv = this.streamingMessage.querySelector('.chat-text');
        if (textDiv) {
            textDiv.innerHTML = Markdown.render(finalContent);
        }
        
        // Add images if present
        if (images.length > 0) {
            const resultsDiv = document.createElement('div');
            resultsDiv.className = 'chat-results';

            images.forEach(img => {
                const card = document.createElement('div');
                card.className = 'chat-result-card';
                card.innerHTML = `
                    <img src="data:image/png;base64,${img.image}" 
                         alt="${Utils.escapeHtml(img.summary || 'Result')}" 
                         onclick="Chat.showImage(this.src)">
                    <div class="chat-result-caption">
                        <span>${Utils.escapeHtml(img.summary || 'Result')}</span>
                        <button class="btn btn-sm btn-ghost" onclick="Utils.downloadBase64Image('${img.image}', 'result.png')">
                            <i class="fas fa-download"></i>
                        </button>
                    </div>
                `;
                resultsDiv.appendChild(card);
            });

            const contentDiv = this.streamingMessage.querySelector('.chat-content');
            contentDiv?.appendChild(resultsDiv);
        }
        
        this.streamingMessage.removeAttribute('id');
        this.streamingMessage = null;
        this.streamingContent = '';
        this.isRealStreaming = false;
    },

    /**
     * Render non-streaming atomic response (no cursor, no animation)
     * Used when provider does not support streaming
     */
    renderAtomicResponse(content, images = []) {
        // Hide typing indicator if still visible
        this.hideTyping();
        
        // Remove any streaming placeholder that was created
        if (this.streamingMessage) {
            this.streamingMessage.remove();
            this.streamingMessage = null;
        }
        this.streamingContent = '';
        this.isRealStreaming = false;
        
        // Directly add the complete message without cursor or animation
        this.addMessage('assistant', content, images);
    },

    /**
     * Show typing indicator (before streaming starts)
     */
    showTyping() {
        const inner = this.ensureMessagesInner();
        if (!inner) return;

        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'chat-message assistant';
        typingDiv.innerHTML = `
            <div class="chat-content">
                <div class="chat-bubble">
                    <div class="typing-indicator">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                </div>
            </div>
        `;

        inner.appendChild(typingDiv);
        this.scrollToBottom();
    },

    /**
     * Hide typing indicator
     */
    hideTyping() {
        document.getElementById('typing-indicator')?.remove();
    },

    /**
     * Scroll chat to bottom (respects auto-scroll setting)
     * @param {boolean} force - Force scroll even if user scrolled up
     */
    scrollToBottom(force = false) {
        if (!force && !this.autoScrollEnabled) return;
        
        const messagesDiv = document.getElementById('chat-messages');
        if (messagesDiv) {
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    },

    /**
     * Enable input area
     */
    enableInput() {
        const inputArea = document.getElementById('chat-input-area');
        if (inputArea) inputArea.classList.remove('hidden');
    },

    /**
     * Disable input area
     */
    disableInput() {
        const inputArea = document.getElementById('chat-input-area');
        if (inputArea) inputArea.classList.add('hidden');
    },

    /**
     * Resize textarea based on content
     */
    resizeTextarea() {
        const textarea = document.getElementById('chat-input');
        if (!textarea) return;

        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    },

    /**
     * Toggle attachment menu
     */
    toggleAttachmentMenu() {
        const menu = document.getElementById('attachment-menu');
        if (menu) {
            menu.classList.toggle('hidden');
        }
    },

    /**
     * Hide attachment menu
     */
    hideAttachmentMenu() {
        const menu = document.getElementById('attachment-menu');
        if (menu) {
            menu.classList.add('hidden');
        }
    },

    /**
     * Handle file selection
     */
    async handleFileSelect(event, type) {
        const files = event.target.files;
        if (!files?.length) return;

        for (const file of files) {
            // Validate
            if (type === 'image' && !file.type.startsWith('image/')) {
                Utils.showToast('Please select an image file', 'error');
                continue;
            }

            if (file.size > 10 * 1024 * 1024) {
                Utils.showToast('File too large (max 10MB)', 'error');
                continue;
            }

            // Add to attachments
            const attachment = {
                id: Utils.generateId(),
                file: file,
                type: type,
                name: file.name,
                size: file.size
            };

            // Get preview for images
            if (type === 'image') {
                attachment.preview = await Utils.readFileAsDataURL(file);
            }

            this.attachments.push(attachment);
        }

        this.renderAttachments();

        // Clear input
        event.target.value = '';
    },

    /**
     * Render attachment previews
     */
    renderAttachments() {
        const preview = document.getElementById('attachment-preview');
        const attachBtn = document.getElementById('attachment-btn');
        if (!preview) return;

        if (this.attachments.length === 0) {
            preview.classList.add('hidden');
            attachBtn?.classList.remove('has-files');
            return;
        }

        preview.classList.remove('hidden');
        attachBtn?.classList.add('has-files');

        preview.innerHTML = this.attachments.map(att => {
            if (att.type === 'image' && att.preview) {
                return `
                    <div class="attachment-item" data-id="${att.id}">
                        <img src="${att.preview}" alt="${Utils.escapeHtml(att.name)}">
                        <span>${Utils.escapeHtml(att.name)}</span>
                        <button class="attachment-remove" onclick="Chat.removeAttachment('${att.id}')">&times;</button>
                    </div>
                `;
            } else {
                const icon = att.name.endsWith('.pdf') ? 'fa-file-pdf' : 'fa-file-alt';
                return `
                    <div class="attachment-item" data-id="${att.id}">
                        <div class="file-icon"><i class="fas ${icon}"></i></div>
                        <span>${Utils.escapeHtml(att.name)}</span>
                        <button class="attachment-remove" onclick="Chat.removeAttachment('${att.id}')">&times;</button>
                    </div>
                `;
            }
        }).join('');
    },

    /**
     * Remove attachment
     */
    removeAttachment(id) {
        this.attachments = this.attachments.filter(a => a.id !== id);
        this.renderAttachments();
    },

    /**
     * Send message with streaming support
     */
    async send() {
        if (this.isLoading) return;

        const textarea = document.getElementById('chat-input');
        const message = textarea?.value.trim() || '';

        // Need either message or attachments
        if (!message && this.attachments.length === 0) return;

        // Ensure we have a session
        if (!this.sessionId) {
            const session = Store.createSession();
            this.sessionId = session.id;
        }

        // Capture current attachments before clearing
        const currentAttachments = [...this.attachments];
        const imageAttachments = currentAttachments.filter(a => a.type === 'image');
        
        // Build display message (text only, images shown as thumbnails)
        let displayMessage = message || (imageAttachments.length > 0 ? 'Analyze this image' : '');

        // Convert attachments to storable format
        const storableAttachments = imageAttachments.map(att => ({
            id: att.id,
            type: att.type,
            name: att.name,
            mimeType: att.file?.type || 'image/png',
            previewUrl: att.preview
        }));

        // Save user message to Store
        const userMessageId = Store.generateId();
        Store.addMessage({
            id: userMessageId,
            role: 'user',
            content: displayMessage,
            attachments: storableAttachments,
            status: 'complete'
        });

        // Add user message to UI with image thumbnails
        this.addMessage('user', displayMessage, [], imageAttachments);

        // Clear input and attachments IMMEDIATELY after displaying user message
        if (textarea) textarea.value = '';
        this.resizeTextarea();
        this.attachments = [];
        this.renderAttachments();
        
        // Reset file inputs
        const imageInput = document.getElementById('attach-image-input');
        const fileInput = document.getElementById('attach-file-input');
        if (imageInput) imageInput.value = '';
        if (fileInput) fileInput.value = '';

        // Create agent message in store with streaming status
        this.streamingMessageId = Store.generateId();
        Store.addMessage({
            id: this.streamingMessageId,
            role: 'agent',
            content: '',
            status: 'streaming'
        });

        // Show typing indicator
        this.showTyping();
        this.setLoading(true);

        // Check if we have image attachments - need to use non-streaming for multipart
        const imageAttachment = imageAttachments[0];

        if (imageAttachment) {
            // Use non-streaming for image uploads
            await this.sendWithImage(message, imageAttachment);
        } else {
            // Use streaming
            await this.sendStreaming(message);
        }
    },

    /**
     * Send message with streaming (SSE)
     */
    async sendStreaming(message) {
        try {
            const response = await fetch('/agent/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            this.hideTyping();
            
            // Create streaming message placeholder (cursor shown only if real tokens arrive)
            this.streamingMessage = this.createStreamingMessage();
            this.streamingContent = '';
            this.pendingTokens = '';
            this.lastRenderTime = 0;
            this.isRealStreaming = false;  // Will be set to true when real tokens arrive
            
            const toolCalls = [];
            
            // Process SSE stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    // Handle both "data: {...}" and "event: xxx\ndata: {...}" formats
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            await this.handleStreamEvent(data, toolCalls);
                        } catch (e) {
                            // Skip malformed events
                        }
                    }
                    // Skip "event:" lines as data follows on next line
                }
            }
            
            // Flush any remaining pending tokens after stream ends
            if (this.pendingTokens) {
                this.streamingContent += this.pendingTokens;
                this.pendingTokens = '';
                this.updateStreamingMessage(this.streamingContent);
            }

        } catch (err) {
            this.hideTyping();
            
            // Update store with error status if we have a streaming message
            if (this.streamingMessageId) {
                Store.updateMessage(this.streamingMessageId, {
                    content: this.streamingContent || '',
                    status: 'error',
                    errorMessage: 'Connection lost: ' + err.message
                });
                this.streamingMessageId = null;
            }
            
            // Clean up streaming message if it exists
            if (this.streamingMessage) {
                this.streamingMessage.remove();
                this.streamingMessage = null;
            }
            
            console.error('Streaming error:', err);
            
            // Fall back to non-streaming
            await this.sendNonStreaming(message);
        } finally {
            this.setLoading(false);
        }
    },

    /**
     * Handle individual stream events
     */
    async handleStreamEvent(data, toolCalls) {
        if (data.session_id) {
            this.sessionId = data.session_id;
        }
        
        if (data.token !== undefined) {
            // Real streaming token - append to pending buffer and trigger incremental render
            // This ensures text appears progressively as tokens arrive
            this.isRealStreaming = true;
            this.appendToken(data.token);
        }
        
        if (data.tool) {
            // Tool call completed
            toolCalls.push(data);
        }
        
        // Handle non-streaming atomic response (streaming: false or complete event)
        if (data.streaming === false || (data.response !== undefined && data.streaming === false)) {
            // Non-streaming mode: render complete response atomically
            // No cursor, no typing animation - just display the full response
            const finalContent = data.response || '';
            
            // Extract images from tool calls
            const resultImages = [];
            for (const tc of (data.tool_calls || toolCalls)) {
                const result = tc.result || {};
                const resultData = result.data || result;
                const imageBase64 = resultData.result_image_base64 ||
                                   resultData.annotated_image ||
                                   resultData.visualization;
                if (imageBase64) {
                    resultImages.push({
                        toolName: tc.name,
                        image: imageBase64,
                        summary: resultData.summary || 'Inference result'
                    });
                }
            }
            
            // Update message in store as complete
            if (this.streamingMessageId) {
                Store.updateMessage(this.streamingMessageId, {
                    content: finalContent,
                    status: 'complete'
                });
            }
            
            // Render complete response without streaming cursor
            this.renderAtomicResponse(finalContent, resultImages);
            this.streamingMessageId = null;
            return;
        }
        
        if (data.response !== undefined || data.success !== undefined) {
            // Done event (streaming completed) - flush any pending tokens first
            if (this.pendingTokens) {
                this.streamingContent += this.pendingTokens;
                this.pendingTokens = '';
            }
            
            // Use final content from server or accumulated content
            const finalContent = data.response || this.streamingContent;
            
            // Extract images from tool calls
            const resultImages = [];
            for (const tc of (data.tool_calls || toolCalls)) {
                const result = tc.result || {};
                const resultData = result.data || result;
                const imageBase64 = resultData.result_image_base64 ||
                                   resultData.annotated_image ||
                                   resultData.visualization;
                if (imageBase64) {
                    resultImages.push({
                        toolName: tc.name,
                        image: imageBase64,
                        summary: resultData.summary || 'Inference result'
                    });
                }
            }
            
            // Update message in store as complete
            if (this.streamingMessageId) {
                Store.updateMessage(this.streamingMessageId, {
                    content: finalContent,
                    status: 'complete'
                });
            }
            
            this.finalizeStreamingMessage(finalContent, resultImages);
            this.streamingMessageId = null;
        }
        
        if (data.error) {
            // Error event - update message in store with error status
            if (this.streamingMessageId) {
                Store.updateMessage(this.streamingMessageId, {
                    content: this.streamingContent || '',
                    status: 'error',
                    errorMessage: data.error
                });
            }
            
            if (this.streamingMessage) {
                this.streamingMessage.remove();
                this.streamingMessage = null;
            }
            this.addMessage('system', data.error);
            this.streamingMessageId = null;
        }
    },

    /**
     * Fallback to non-streaming request
     */
    async sendNonStreaming(message) {
        try {
            const response = await fetch('/agent/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId
                })
            });

            const data = await response.json();
            this.hideTyping();

            if (data.success) {
                this.sessionId = data.session_id;

                // Extract images from tool calls
                let resultImages = [];
                if (data.tool_calls?.length > 0) {
                    data.tool_calls.forEach(tc => {
                        if (tc.result?.data) {
                            const resultData = tc.result.data;
                            const imageBase64 = resultData.result_image_base64 ||
                                               resultData.annotated_image ||
                                               resultData.visualization;
                            if (imageBase64) {
                                resultImages.push({
                                    toolName: tc.name,
                                    image: imageBase64,
                                    summary: resultData.summary || 'Inference result'
                                });
                            }
                        }
                    });
                }

                // Update message in store
                if (this.streamingMessageId) {
                    Store.updateMessage(this.streamingMessageId, {
                        content: data.response,
                        status: 'complete'
                    });
                }

                this.addMessage('assistant', data.response, resultImages);
                this.streamingMessageId = null;
            } else {
                // Update message in store with error
                if (this.streamingMessageId) {
                    Store.updateMessage(this.streamingMessageId, {
                        content: '',
                        status: 'error',
                        errorMessage: data.response || data.error || 'An error occurred'
                    });
                }
                this.addMessage('system', data.response || data.error || 'An error occurred');
                this.streamingMessageId = null;
            }
        } catch (err) {
            this.hideTyping();
            console.error('Chat error:', err);
            
            // Update message in store with error
            if (this.streamingMessageId) {
                Store.updateMessage(this.streamingMessageId, {
                    content: '',
                    status: 'error',
                    errorMessage: 'Failed to send message: ' + err.message
                });
            }
            this.addMessage('system', 'Failed to send message: ' + err.message);
            this.streamingMessageId = null;
        }
    },

    /**
     * Send with image attachment (non-streaming, multipart)
     */
    async sendWithImage(message, imageAttachment) {
        try {
            const formData = new FormData();
            formData.append('message', message || 'Please analyze this image');
            formData.append('image', imageAttachment.file);
            if (this.sessionId) {
                formData.append('session_id', this.sessionId);
            }

            const response = await fetch('/agent/chat', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            this.hideTyping();

            if (data.success) {
                this.sessionId = data.session_id;

                // Extract images from tool calls
                let resultImages = [];
                if (data.tool_calls?.length > 0) {
                    data.tool_calls.forEach(tc => {
                        if (tc.result?.data) {
                            const resultData = tc.result.data;
                            const imageBase64 = resultData.result_image_base64 ||
                                               resultData.annotated_image ||
                                               resultData.visualization;
                            if (imageBase64) {
                                resultImages.push({
                                    toolName: tc.name,
                                    image: imageBase64,
                                    summary: resultData.summary || 'Inference result'
                                });
                            }
                        }
                    });
                }

                // Update message in store
                if (this.streamingMessageId) {
                    Store.updateMessage(this.streamingMessageId, {
                        content: data.response,
                        status: 'complete'
                    });
                }

                this.addMessage('assistant', data.response, resultImages);
                this.streamingMessageId = null;
            } else {
                // Update message in store with error
                if (this.streamingMessageId) {
                    Store.updateMessage(this.streamingMessageId, {
                        content: '',
                        status: 'error',
                        errorMessage: data.response || data.error || 'An error occurred'
                    });
                }
                this.addMessage('system', data.response || data.error || 'An error occurred');
                this.streamingMessageId = null;
            }
        } catch (err) {
            this.hideTyping();
            console.error('Image upload error:', err);
            
            // Update message in store with error
            if (this.streamingMessageId) {
                Store.updateMessage(this.streamingMessageId, {
                    content: '',
                    status: 'error',
                    errorMessage: 'Failed to upload image: ' + err.message
                });
            }
            this.addMessage('system', 'Failed to upload image: ' + err.message);
            this.streamingMessageId = null;
        } finally {
            this.setLoading(false);
        }
    },

    /**
     * Set loading state
     */
    setLoading(loading) {
        this.isLoading = loading;
        Store.setStreaming(loading);
        const sendBtn = document.getElementById('chat-send-btn');
        if (sendBtn) {
            sendBtn.disabled = loading;
        }
    },

    /**
     * Show image in modal
     */
    showImage(src) {
        const modal = document.getElementById('image-modal');
        const img = document.getElementById('modal-image');
        if (modal && img) {
            img.src = src;
            modal.classList.remove('hidden');
        }
    },

    /**
     * Close image modal
     */
    closeImageModal() {
        const modal = document.getElementById('image-modal');
        if (modal) {
            modal.classList.add('hidden');
        }
    }
};

// Export
window.Chat = Chat;
