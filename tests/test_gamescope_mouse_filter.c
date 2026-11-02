/* Manual SDL2 regression test. Run under dummy video driver with LD_PRELOAD
 * pointing to the Gamescope no-host-mouse shim. No display or Minecraft needed.
 */
#include <SDL2/SDL.h>
#include <stdio.h>

int main(void)
{
    SDL_Event e = {0};
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        return 1;
    SDL_Window *window = SDL_CreateWindow("shim test", 0, 0, 100, 100, SDL_WINDOW_HIDDEN);
    if (!window)
        return 2;
    SDL_FlushEvents(SDL_FIRSTEVENT, SDL_LASTEVENT);

    if (SDL_SetRelativeMouseMode(SDL_TRUE) != 0 || SDL_GetRelativeMouseMode() != SDL_FALSE)
        return 3; /* Never lock host pointer. */

    e.type = SDL_MOUSEMOTION; e.motion.xrel = 42;
    if (SDL_PushEvent(&e) != 1) return 4;
    e.type = SDL_MOUSEBUTTONDOWN; e.button.button = SDL_BUTTON_LEFT;
    if (SDL_PushEvent(&e) != 1) return 5;
    e.type = SDL_MOUSEWHEEL; e.wheel.y = 1;
    if (SDL_PushEvent(&e) != 1) return 6;
    e.type = SDL_KEYDOWN; e.key.keysym.sym = SDLK_a;
    if (SDL_PushEvent(&e) != 1) return 7;
    e.type = SDL_QUIT;
    if (SDL_PushEvent(&e) != 1) return 8;

    if (SDL_WaitEvent(&e) != 1 || e.type != SDL_KEYDOWN || e.key.keysym.sym != SDLK_a)
        return 9; /* Mouse events must be discarded, keyboard still works. */
    if (SDL_WaitEvent(&e) != 1 || e.type != SDL_QUIT)
        return 10; /* Normal window control must continue. */

    SDL_DestroyWindow(window);
    SDL_Quit();
    puts("PASS: pointer stays free; host motion/buttons/wheel blocked; keyboard/quit preserved");
    return 0;
}
