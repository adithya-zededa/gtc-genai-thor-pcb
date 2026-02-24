/**
 * Utility Functions
 * Shared helpers used across the application
 */

const Utils = {
    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        if (text === null || text === undefined) return '';
        const div = document.createElement('div');
        div.textContent = String(text);
        return div.innerHTML;
    },

    /**
     * Generate unique ID
     */
    generateId() {
        return `id_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    },

    /**
     * Debounce function calls
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    },

    /**
     * Format timestamp
     */
    formatTime(date) {
        return new Intl.DateTimeFormat('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        }).format(date);
    },

    /**
     * Copy text to clipboard
     * Uses modern Clipboard API with fallback for non-secure contexts
     */
    async copyToClipboard(text) {
        // Try modern Clipboard API first (requires secure context)
        if (navigator.clipboard && window.isSecureContext) {
            try {
                await navigator.clipboard.writeText(text);
                return true;
            } catch (err) {
                console.warn('Clipboard API failed, trying fallback:', err);
            }
        }
        
        // Fallback for non-secure contexts (HTTP)
        try {
            const textArea = document.createElement('textarea');
            textArea.value = text;
            
            // Avoid scrolling to bottom
            textArea.style.top = '0';
            textArea.style.left = '0';
            textArea.style.position = 'fixed';
            textArea.style.opacity = '0';
            textArea.style.pointerEvents = 'none';
            
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            const success = document.execCommand('copy');
            document.body.removeChild(textArea);
            
            if (success) {
                return true;
            }
        } catch (err) {
            console.error('Fallback copy failed:', err);
        }
        
        return false;
    },

    /**
     * Download file from data
     */
    downloadFile(data, filename, mimeType = 'application/json') {
        const blob = new Blob([data], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },

    /**
     * Download image from base64
     */
    downloadBase64Image(base64Data, filename = 'image.png') {
        const link = document.createElement('a');
        link.href = `data:image/png;base64,${base64Data}`;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    },

    /**
     * Read file as text
     */
    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(e);
            reader.readAsText(file);
        });
    },

    /**
     * Read file as data URL (for images)
     */
    readFileAsDataURL(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(e);
            reader.readAsDataURL(file);
        });
    },

    /**
     * Validate base64 string
     */
    isValidBase64(str) {
        if (!str || typeof str !== 'string') return false;
        const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/;
        const sanitized = str.replace(/[^A-Za-z0-9+/=]/g, '');
        return base64Regex.test(sanitized);
    },

    /**
     * Create safe base64 image element
     */
    createBase64Image(base64Data, alt = 'Image') {
        if (!this.isValidBase64(base64Data)) return null;
        const img = document.createElement('img');
        img.src = `data:image/jpeg;base64,${base64Data}`;
        img.alt = alt;
        return img;
    },

    /**
     * Simple element creation helper
     */
    createElement(tag, attributes = {}, children = []) {
        const el = document.createElement(tag);
        
        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'className') {
                el.className = value;
            } else if (key === 'innerHTML') {
                el.innerHTML = value;
            } else if (key === 'textContent') {
                el.textContent = value;
            } else if (key.startsWith('on') && typeof value === 'function') {
                el.addEventListener(key.slice(2).toLowerCase(), value);
            } else {
                el.setAttribute(key, value);
            }
        });

        children.forEach(child => {
            if (typeof child === 'string') {
                el.appendChild(document.createTextNode(child));
            } else if (child instanceof Node) {
                el.appendChild(child);
            }
        });

        return el;
    },

    /**
     * Show toast notification
     */
    showToast(message, type = 'info', duration = 3000) {
        const toast = this.createElement('div', {
            className: `toast toast-${type}`,
            textContent: message
        });
        
        const container = document.getElementById('toast-container') || (() => {
            const c = this.createElement('div', { id: 'toast-container' });
            document.body.appendChild(c);
            return c;
        })();
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('toast-fade-out');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
};

// Export for use in other modules
window.Utils = Utils;
