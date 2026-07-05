(function () {
    const theme = localStorage.getItem('theme');
    if (!theme) {
        const prefersDark = window.matchMedia(
            '(prefers-color-scheme: dark)',
        ).matches;
        document.documentElement.setAttribute(
            'data-theme',
            prefersDark ? 'dark' : 'light',
        );
    } else {
        document.documentElement.setAttribute('data-theme', theme);
    }
})();
