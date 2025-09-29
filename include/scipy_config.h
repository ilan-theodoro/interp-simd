#ifndef SCIPY_CONFIG_H
#define SCIPY_CONFIG_H

#if defined(__cplusplus)
#define SCIPY_TLS thread_local
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define SCIPY_TLS _Thread_local
#elif defined(_MSC_VER)
#define SCIPY_TLS __declspec(thread)
#elif defined(__GNUC__)
#define SCIPY_TLS __thread
#else
#define SCIPY_TLS
#endif

#endif /* SCIPY_CONFIG_H */
