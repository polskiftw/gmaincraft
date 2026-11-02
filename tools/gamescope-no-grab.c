/*
 * Gamescope SDL2 backend: leave the real desktop pointer free and prevent its
 * mouse input from reaching the nested Minecraft client.
 *
 * The shim is loaded into Gamescope only. The MCio Python controller and
 * Minecraft JVM explicitly run with LD_PRELOAD removed. The agent's MCio
 * actions still reach Minecraft through MCio's internal input protocol.
 */
#define _GNU_SOURCE
#include <SDL2/SDL.h>
#include <dlfcn.h>
#include <stdio.h>

static void report_once(const char *message, int *seen)
{
    if (!*seen) {
        fputs(message, stderr);
        *seen = 1;
    }
}

int SDL_SetRelativeMouseMode(SDL_bool enabled)
{
    static int noted = 0;
    if (enabled) {
        report_once("[mcio-no-grab] Blocking Gamescope pointer lock.\n", &noted);
        return 0;
    }
    int (*next)(SDL_bool) = dlsym(RTLD_NEXT, "SDL_SetRelativeMouseMode");
    return next ? next(SDL_FALSE) : 0;
}

int SDL_WaitEvent(SDL_Event *event)
{
    int (*next)(SDL_Event *) = dlsym(RTLD_NEXT, "SDL_WaitEvent");
    static int noted = 0;
    if (!next)
        return 0;
    if (!event)
        return next(event);

    for (;;) {
        int result = next(event);
        if (result != 1)
            return result;
        switch (event->type) {
        case SDL_MOUSEMOTION:
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
        case SDL_MOUSEWHEEL:
            report_once("[mcio-no-grab] Discarding host mouse motion/click/scroll events.\n", &noted);
            continue;  /* Keep waiting until a non-mouse event arrives. */
        default:
            return result;
        }
    }
}
