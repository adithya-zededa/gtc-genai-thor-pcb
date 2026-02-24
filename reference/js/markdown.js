/**
 * Markdown Renderer
 * Converts markdown to HTML with syntax highlighting for code blocks
 */

const Markdown = {
    /**
     * Generate unique ID for collapsible sections
     */
    _thinkingCounter: 0,
    
    /**
     * Render markdown text to HTML
     */
    render(text) {
        if (!text) return '';

        // Store code blocks to protect them
        const codeBlocks = [];
        const inlineCodes = [];
        const thinkingBlocks = [];

        // First, handle <think> tags - extract them before other processing
        let html = text.replace(/<think>([\s\S]*?)<\/think>/gi, (match, content) => {
            const placeholder = `%%THINKING${thinkingBlocks.length}%%`;
            thinkingBlocks.push(content.trim());
            return placeholder;
        });

        // Handle CODEBLOCK placeholders that LLMs sometimes output literally
        // Convert CODEBLOCK0, CODEBLOCK1_, etc. to a placeholder note
        html = html.replace(/CODEBLOCK\d+_?/g, '<em class="code-placeholder">[Code example - see documentation]</em>');

        // Extract fenced code blocks
        html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (match, lang, code) => {
            const placeholder = `%%CODEBLOCK${codeBlocks.length}%%`;
            codeBlocks.push({ lang: lang || 'text', code: code.trim() });
            return placeholder;
        });

        // Extract inline code
        html = html.replace(/`([^`]+)`/g, (match, code) => {
            const placeholder = `%%INLINECODE${inlineCodes.length}%%`;
            inlineCodes.push(code);
            return placeholder;
        });

        // Escape HTML in remaining content
        html = Utils.escapeHtml(html);

        // Restore inline code
        inlineCodes.forEach((code, i) => {
            html = html.replace(
                `%%INLINECODE${i}%%`,
                `<code>${Utils.escapeHtml(code)}</code>`
            );
        });

        // Headers
        html = html.replace(/^###### (.+)$/gm, '<h6>$1</h6>');
        html = html.replace(/^##### (.+)$/gm, '<h5>$1</h5>');
        html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        // Bold and italic
        html = html.replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        html = html.replace(/___([^_]+)___/g, '<strong><em>$1</em></strong>');
        html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
        html = html.replace(/_([^_]+)_/g, '<em>$1</em>');

        // Links
        html = html.replace(
            /\[([^\]]+)\]\(([^)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
        );

        // Horizontal rules
        html = html.replace(/^---$/gm, '<hr>');
        html = html.replace(/^\*\*\*$/gm, '<hr>');

        // Blockquotes
        html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
        html = html.replace(/<\/blockquote>\n<blockquote>/g, '<br>');

        // Lists - unordered
        html = html.replace(/^(\s*)[-*] (.+)$/gm, '$1<li>$2</li>');
        
        // Lists - ordered
        html = html.replace(/^(\s*)\d+\. (.+)$/gm, '$1<li>$2</li>');

        // Wrap consecutive list items
        html = html.replace(/(<li>[\s\S]*?<\/li>)(?=\s*<li>)/g, '$1');
        html = html.replace(/(<li>[\s\S]*?<\/li>)+/g, (match) => {
            // Determine if ordered or unordered based on original text
            return `<ul>${match}</ul>`;
        });

        // Clean up list formatting
        html = html.replace(/<\/li>\s*<li>/g, '</li><li>');
        html = html.replace(/<\/ul>\s*<ul>/g, '');

        // Paragraphs
        html = html.replace(/\n\n+/g, '</p><p>');
        html = html.replace(/([^>])\n([^<])/g, '$1<br>$2');

        // Wrap content if not starting with block element
        if (!html.match(/^<(h[1-6]|ul|ol|pre|blockquote|hr|p)/)) {
            html = '<p>' + html + '</p>';
        }

        // Clean up
        html = html.replace(/<p>\s*<\/p>/g, '');
        html = html.replace(/<p>(<(h[1-6]|ul|ol|pre|blockquote|hr))/g, '$1');
        html = html.replace(/(<\/(h[1-6]|ul|ol|pre|blockquote|hr)>)<\/p>/g, '$1');

        // Restore code blocks with syntax highlighting
        codeBlocks.forEach((block, i) => {
            const highlighted = this.highlightCode(block.code, block.lang);
            const codeHtml = `
                <div class="code-block">
                    <div class="code-header">
                        <span class="code-lang">${Utils.escapeHtml(block.lang)}</span>
                        <button class="code-copy-btn" onclick="Markdown.copyCode(this)" data-code="${Utils.escapeHtml(block.code).replace(/"/g, '&quot;')}">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                    </div>
                    <pre><code class="language-${block.lang}">${highlighted}</code></pre>
                </div>
            `;
            html = html.replace(`%%CODEBLOCK${i}%%`, codeHtml);
        });

        // Restore thinking blocks as collapsible sections
        thinkingBlocks.forEach((content, i) => {
            const thinkingId = `thinking-${Date.now()}-${this._thinkingCounter++}`;
            const thinkingHtml = `
                <details class="thinking-block">
                    <summary class="thinking-toggle">
                        <i class="fas fa-brain"></i>
                        <span>Reasoning</span>
                        <i class="fas fa-chevron-down toggle-icon"></i>
                    </summary>
                    <div class="thinking-content">${Utils.escapeHtml(content).replace(/\n/g, '<br>')}</div>
                </details>
            `;
            html = html.replace(`%%THINKING${i}%%`, thinkingHtml);
        });

        return html;
    },

    /**
     * Basic syntax highlighting
     */
    highlightCode(code, lang) {
        // Escape HTML first
        let escaped = Utils.escapeHtml(code);

        // Apply language-specific highlighting
        switch (lang.toLowerCase()) {
            case 'javascript':
            case 'js':
                escaped = this.highlightJS(escaped);
                break;
            case 'python':
            case 'py':
                escaped = this.highlightPython(escaped);
                break;
            case 'json':
                escaped = this.highlightJSON(escaped);
                break;
            case 'bash':
            case 'shell':
            case 'sh':
                escaped = this.highlightBash(escaped);
                break;
            case 'html':
            case 'xml':
                escaped = this.highlightHTML(escaped);
                break;
            case 'css':
                escaped = this.highlightCSS(escaped);
                break;
            default:
                // No highlighting for unknown languages
                break;
        }

        return escaped;
    },

    /**
     * JavaScript syntax highlighting
     */
    highlightJS(code) {
        // Keywords
        code = code.replace(
            /\b(const|let|var|function|return|if|else|for|while|do|switch|case|break|continue|new|this|class|extends|import|export|from|default|async|await|try|catch|throw|finally|typeof|instanceof)\b/g,
            '<span class="hl-keyword">$1</span>'
        );
        
        // Strings
        code = code.replace(
            /(&quot;[^&]*&quot;|&#39;[^&]*&#39;|`[^`]*`)/g,
            '<span class="hl-string">$1</span>'
        );
        
        // Numbers
        code = code.replace(
            /\b(\d+\.?\d*)\b/g,
            '<span class="hl-number">$1</span>'
        );
        
        // Comments
        code = code.replace(
            /(\/\/.*$)/gm,
            '<span class="hl-comment">$1</span>'
        );
        
        // Functions
        code = code.replace(
            /\b([a-zA-Z_]\w*)\s*\(/g,
            '<span class="hl-function">$1</span>('
        );

        return code;
    },

    /**
     * Python syntax highlighting
     */
    highlightPython(code) {
        // Keywords
        code = code.replace(
            /\b(def|class|return|if|elif|else|for|while|import|from|as|try|except|raise|finally|with|lambda|yield|pass|break|continue|and|or|not|in|is|None|True|False|self|async|await)\b/g,
            '<span class="hl-keyword">$1</span>'
        );
        
        // Strings
        code = code.replace(
            /(&quot;&quot;&quot;[\s\S]*?&quot;&quot;&quot;|&#39;&#39;&#39;[\s\S]*?&#39;&#39;&#39;|&quot;[^&]*&quot;|&#39;[^&]*&#39;)/g,
            '<span class="hl-string">$1</span>'
        );
        
        // Numbers
        code = code.replace(
            /\b(\d+\.?\d*)\b/g,
            '<span class="hl-number">$1</span>'
        );
        
        // Comments
        code = code.replace(
            /(#.*$)/gm,
            '<span class="hl-comment">$1</span>'
        );
        
        // Functions/methods
        code = code.replace(
            /\b([a-zA-Z_]\w*)\s*\(/g,
            '<span class="hl-function">$1</span>('
        );
        
        // Decorators
        code = code.replace(
            /(@\w+)/g,
            '<span class="hl-decorator">$1</span>'
        );

        return code;
    },

    /**
     * JSON syntax highlighting
     */
    highlightJSON(code) {
        // Property names
        code = code.replace(
            /(&quot;[^&]+&quot;)\s*:/g,
            '<span class="hl-property">$1</span>:'
        );
        
        // String values
        code = code.replace(
            /:\s*(&quot;[^&]*&quot;)/g,
            ': <span class="hl-string">$1</span>'
        );
        
        // Numbers
        code = code.replace(
            /:\s*(\d+\.?\d*)/g,
            ': <span class="hl-number">$1</span>'
        );
        
        // Booleans and null
        code = code.replace(
            /:\s*(true|false|null)/g,
            ': <span class="hl-keyword">$1</span>'
        );

        return code;
    },

    /**
     * Bash/Shell syntax highlighting
     */
    highlightBash(code) {
        // Comments
        code = code.replace(
            /(#.*$)/gm,
            '<span class="hl-comment">$1</span>'
        );
        
        // Strings
        code = code.replace(
            /(&quot;[^&]*&quot;|&#39;[^&]*&#39;)/g,
            '<span class="hl-string">$1</span>'
        );
        
        // Variables
        code = code.replace(
            /(\$\w+|\$\{[^}]+\})/g,
            '<span class="hl-variable">$1</span>'
        );
        
        // Commands at start of line
        code = code.replace(
            /^(\s*)(curl|wget|git|npm|pip|python|node|docker|kubectl|make|cd|ls|cat|echo|export|source)/gm,
            '$1<span class="hl-command">$2</span>'
        );

        return code;
    },

    /**
     * HTML syntax highlighting
     */
    highlightHTML(code) {
        // Tags
        code = code.replace(
            /(&lt;\/?)([\w-]+)/g,
            '$1<span class="hl-tag">$2</span>'
        );
        
        // Attributes
        code = code.replace(
            /\s([\w-]+)=/g,
            ' <span class="hl-attribute">$1</span>='
        );
        
        // Attribute values
        code = code.replace(
            /=(&quot;[^&]*&quot;)/g,
            '=<span class="hl-string">$1</span>'
        );

        return code;
    },

    /**
     * CSS syntax highlighting
     */
    highlightCSS(code) {
        // Selectors
        code = code.replace(
            /^([^{]+)\{/gm,
            '<span class="hl-selector">$1</span>{'
        );
        
        // Properties
        code = code.replace(
            /([\w-]+)\s*:/g,
            '<span class="hl-property">$1</span>:'
        );
        
        // Values with units
        code = code.replace(
            /:\s*(\d+(?:\.\d+)?(?:px|em|rem|%|vh|vw|s|ms))/g,
            ': <span class="hl-number">$1</span>'
        );
        
        // Colors
        code = code.replace(
            /(#[0-9a-fA-F]{3,8})/g,
            '<span class="hl-color">$1</span>'
        );

        return code;
    },

    /**
     * Copy code to clipboard
     */
    async copyCode(button) {
        const code = button.dataset.code
            .replace(/&quot;/g, '"')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&amp;/g, '&');
        
        const success = await Utils.copyToClipboard(code);
        
        if (success) {
            const icon = button.querySelector('i');
            const originalClass = icon.className;
            
            icon.className = 'fas fa-check';
            button.classList.add('copied');
            
            setTimeout(() => {
                icon.className = originalClass;
                button.classList.remove('copied');
            }, 2000);
        }
    }
};

// Export for use in other modules
window.Markdown = Markdown;
