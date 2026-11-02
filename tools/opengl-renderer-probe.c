/* Inspect the GLX context inside the same nested Gamescope Xwayland display
 * used by Minecraft. This probes the renderer without changing Gentoo. */
#include <stdio.h>
#include <string.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

int main(void) {
    Display *display = XOpenDisplay(NULL);
    if (!display) {
        fputs("FAIL: Could not open nested X display\n",stderr);
        return 1;
    }
    int attrs[] = {GLX_RGBA, None};
    XVisualInfo *visual = glXChooseVisual(display, DefaultScreen(display), attrs);
    if (!visual) {
        fputs("FAIL: No GLX RGB visual available\n",stderr);
        XCloseDisplay(display);
        return 1;
    }
    GLXContext context = glXCreateContext(display, visual, NULL, True);
    Window window = XCreateSimpleWindow(display, RootWindow(display, visual->screen),
                                        0, 0, 64, 64, 0, 0, 0);
    if (!context || !glXMakeCurrent(display, window, context)) {
        fputs("FAIL: Could not create OpenGL context\n",stderr);
        if (context) glXDestroyContext(display,context);
        XDestroyWindow(display,window);
        XFree(visual);
        XCloseDisplay(display);
        return 1;
    }
    const char *vendor = (const char*)glGetString(GL_VENDOR);
    const char *renderer = (const char*)glGetString(GL_RENDERER);
    const char *version = (const char*)glGetString(GL_VERSION);
    printf("OPENGL_VENDOR=%s\nOPENGL_RENDERER=%s\nOPENGL_VERSION=%s\n",
           vendor ? vendor : "unknown", renderer ? renderer : "unknown",
           version ? version : "unknown");
    /* glGetString pointers cease to be valid after deleting the GL context. */
    int software = (!vendor || !renderer || !version ||
                    strstr(renderer,"llvmpipe") || strstr(renderer,"softpipe"));
    fflush(stdout);
    glXMakeCurrent(display, None, NULL);
    glXDestroyContext(display,context);
    XDestroyWindow(display,window);
    XFree(visual);
    XCloseDisplay(display);
    return software ? 1 : 0;
}
