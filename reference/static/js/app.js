/**
 * Main Application Entry Point
 * Initializes all modules on page load
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize state management first
    Store.init();
    
    // Initialize UI modules
    Settings.init();
    Sidebar.init();
    Chat.init();

    console.log('ZEDEDA AI Model Explorer initialized');
});
