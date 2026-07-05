const CACHE = 'tiz-v5';
const STATIC_FILES = [
    'index.html',
    'style.css',
    'app.js',
    'theme.js',
    'favicon.svg',
    'favicon.ico',
    'favicon-192x192.png',
    'favicon-512x512.png',
    'apple-touch-icon.png',
    'manifest.json',
];
const OFFLINE_URL = 'index.html';
const NETWORK_TIMEOUT_MS = 4000;

function getBaseFromUrl() {
    const path = self.location.pathname;
    const idx = path.lastIndexOf('/');
    return idx >= 0 ? path.substring(0, idx + 1) : '/';
}

const BASE = getBaseFromUrl();

const STATIC_ASSET_PATHS = new Set(STATIC_FILES.map((f) => BASE + f));

self.addEventListener('install', (event) => {
    event.waitUntil(
        (async () => {
            const cache = await caches.open(CACHE);
            try {
                await cache.addAll(STATIC_FILES);
            } catch (e) {
                console.warn('Cache addAll failed:', e);
                throw e;
            }
        })(),
    );
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        (async () => {
            const keys = await caches.keys();
            await Promise.all(
                keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)),
            );
            await self.clients.claim();
        })(),
    );
});

function fetchWithTimeout(request) {
    return new Promise((resolve, reject) => {
        const controller = new AbortController();
        const timeoutId = setTimeout(
            () => controller.abort(),
            NETWORK_TIMEOUT_MS,
        );
        fetch(request, { signal: controller.signal })
            .then((response) => {
                clearTimeout(timeoutId);
                resolve(response);
            })
            .catch((err) => {
                clearTimeout(timeoutId);
                reject(err);
            });
    });
}

self.addEventListener('fetch', (event) => {
    const { request } = event;
    if (request.method !== 'GET') return;

    const url = new URL(request.url);

    if (url.origin !== self.location.origin) return;
    if (url.protocol === 'ws:' || url.protocol === 'wss:') return;

    const apiPrefix = BASE + 'api/';
    const isApi = url.pathname.startsWith(apiPrefix);
    if (isApi) {
        event.respondWith(fetch(request));
        return;
    }

    if (request.mode === 'navigate') {
        event.respondWith(
            (async () => {
                try {
                    const response = await fetchWithTimeout(request);
                    if (response.ok) {
                        const cache = await caches.open(CACHE);
                        cache.put(request, response.clone()).catch((e) => {
                            console.warn('Cache put failed:', e);
                        });
                    }
                    return response;
                } catch (e) {
                    const cached = await caches.match(request);
                    if (cached) return cached;
                    const offline = await caches.match(OFFLINE_URL);
                    if (offline) return offline;
                    return new Response('Offline', {
                        status: 503,
                        headers: { 'Content-Type': 'text/plain' },
                    });
                }
            })(),
        );
        return;
    }

    if (STATIC_ASSET_PATHS.has(url.pathname)) {
        event.respondWith(
            (async () => {
                const cached = await caches.match(request);
                if (cached) {
                    fetchWithTimeout(request)
                        .then((response) => {
                            if (response.ok) {
                                caches.open(CACHE).then((cache) => {
                                    cache
                                        .put(request, response.clone())
                                        .catch((e) => {
                                            console.warn(
                                                'Cache put failed:',
                                                e,
                                            );
                                        });
                                });
                            }
                        })
                        .catch(() => {});
                    return cached;
                }
                try {
                    return await fetchWithTimeout(request);
                } catch (e) {
                    return new Response('Offline', {
                        status: 503,
                        headers: { 'Content-Type': 'text/plain' },
                    });
                }
            })(),
        );
        return;
    }

    event.respondWith(
        (async () => {
            const cached = await caches.match(request);
            const networkPromise = fetchWithTimeout(request)
                .then((response) => {
                    if (response.ok) {
                        const cache = caches.open(CACHE);
                        cache.then((c) => {
                            c.put(request, response.clone()).catch((e) => {
                                console.warn('Cache put failed:', e);
                            });
                        });
                    }
                    return response;
                })
                .catch(() => null);

            if (cached) {
                networkPromise.then(() => {});
                return cached;
            }

            const networkResponse = await networkPromise;
            if (networkResponse) return networkResponse;

            return new Response('Offline', {
                status: 503,
                headers: { 'Content-Type': 'text/plain' },
            });
        })(),
    );
});

self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});
