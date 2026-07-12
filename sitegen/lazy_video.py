# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Viewport-driven video loading for the static review pages.

Both review pages (vio_filter, yaw_review) stack 70-100 episode clips of
0.6-1.6 Mbps H.264 on one scroll. `preload="none"` meant every clip started
its network fetch only on the reviewer's click, so playback stuttered while
the buffer caught up; `preload="auto"` on that many tags would instead fire
~100 parallel fetches and starve the one clip actually on screen.

This module emits the middle ground: no `src` in the markup at all (so the
page costs nothing on load), and a script that attaches `src` a screen or two
*before* a clip scrolls into view, at most `MAX_PARALLEL` fetches at a time,
autoplays it (muted) once it is on screen, pauses it when it leaves, and drops
the buffer of clips that scrolled far away so a long page cannot accumulate
100 buffered videos worth of memory.

Anything on screen jumps the fetch queue: the reviewer's current clip must
never wait behind a prefetch for a clip they have not reached yet.
"""

from __future__ import annotations

import html

# ~1.5 screens of lead time: at typical scroll speed that is enough for a
# 3-8 MB clip to buffer its opening seconds before it is looked at, and it
# bounds how many clips can be attached at once (KEEP below).
_PREFETCH_MARGIN = "1200px 0px"
_KEEP = 6  # attached (buffer-holding) videos; off-screen surplus is dropped
_MAX_PARALLEL = 2  # simultaneous fetches, so the on-screen clip owns the pipe
_PLAY_THRESHOLD = 0.35  # fraction visible before autoplay starts
_EGO2_W, _EGO2_H = 590, 480  # ego2_4x.mp4 render size — reserves the layout box

LAZY_VIDEO_SCRIPT = f"""
<script>
(function () {{
  var PREFETCH_MARGIN = "{_PREFETCH_MARGIN}";
  var KEEP = {_KEEP};
  var MAX_PARALLEL = {_MAX_PARALLEL};
  var PLAY_THRESHOLD = {_PLAY_THRESHOLD};

  var videos = [].slice.call(document.querySelectorAll("video[data-src]"));
  if (!videos.length || !("IntersectionObserver" in window)) {{
    videos.forEach(function (v) {{ v.src = v.dataset.src; }});
    return;
  }}
  var reduceMotion = window.matchMedia
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var attached = [];   // videos holding an src, oldest first
  var pending = [];    // videos waiting for a fetch slot
  var inflight = 0;

  function detach(v) {{
    var i = attached.indexOf(v);
    if (i >= 0) attached.splice(i, 1);
    if (v.dataset.state === "loading") inflight -= 1;
    v.pause();
    v.removeAttribute("src");
    v.load();          // without this the buffer survives the src removal
    v.dataset.state = "";
  }}

  function evict() {{
    // Never evict what the reviewer is looking at, even above KEEP.
    for (var i = 0; i < attached.length && attached.length > KEEP; i++) {{
      if (!attached[i].dataset.visible && attached[i].dataset.state === "ready") {{
        detach(attached[i]);
        i -= 1;
      }}
    }}
  }}

  function pump() {{
    while (inflight < MAX_PARALLEL && pending.length) {{
      var v = pending.shift();
      if (v.dataset.state !== "queued") continue;
      inflight += 1;
      v.dataset.state = "loading";
      v.preload = "auto";
      v.src = v.dataset.src;
      v.load();
      attached.push(v);
      evict();
    }}
  }}

  function release(v) {{
    if (v.dataset.state !== "loading") return;
    v.dataset.state = "ready";
    inflight -= 1;
    pump();
  }}

  function request(v) {{
    if (v.dataset.state) return;   // queued | loading | ready
    v.dataset.state = "queued";
    pending.push(v);
    pump();
  }}

  function prioritize(v) {{
    var i = pending.indexOf(v);
    if (i > 0) {{
      pending.splice(i, 1);
      pending.unshift(v);
    }}
  }}

  function play(v) {{
    if (reduceMotion) return;
    var p = v.play();
    if (p && p.catch) p.catch(function () {{}});   // src not attached yet
  }}

  var prefetch = new IntersectionObserver(function (entries) {{
    entries.forEach(function (e) {{
      var v = e.target;
      if (e.isIntersecting) {{
        request(v);
      }} else if (v.dataset.state === "queued") {{
        var i = pending.indexOf(v);
        if (i >= 0) pending.splice(i, 1);
        v.dataset.state = "";
      }} else if (v.dataset.state) {{
        detach(v);
      }}
    }});
  }}, {{rootMargin: PREFETCH_MARGIN}});

  var viewport = new IntersectionObserver(function (entries) {{
    entries.forEach(function (e) {{
      var v = e.target;
      if (e.isIntersecting) {{
        v.dataset.visible = "1";
        request(v);
        prioritize(v);
        play(v);
      }} else {{
        delete v.dataset.visible;
        v.pause();
      }}
    }});
  }}, {{threshold: PLAY_THRESHOLD}});

  videos.forEach(function (v) {{
    v.dataset.state = "";
    // A clip that finished buffering must hand its fetch slot to the next one;
    // "suspend" covers the browser deciding it has buffered enough, "error"
    // covers a missing file (a dead slot would stall every later prefetch).
    ["canplaythrough", "suspend", "error"].forEach(function (evt) {{
      v.addEventListener(evt, function () {{ release(v); }});
    }});
    // Autoplay is retried here because the viewport observer usually fires
    // before the src is attached, so its play() call is rejected.
    v.addEventListener("loadeddata", function () {{
      if (v.dataset.visible) play(v);
    }});
    prefetch.observe(v);
    viewport.observe(v);
  }});
}})();
</script>
"""


def video_tag(src: str, *, style: str = "", cls: str = "",
              width: int = _EGO2_W, height: int = _EGO2_H) -> str:
    """A clip the script above owns: `data-src` (not `src`) so nothing is
    fetched until it is near the viewport, muted/playsinline so the autoplay
    is allowed by browser policy, loop so a 40 s clip keeps replaying while
    the reviewer studies the plots beside it, controls so they can still
    scrub/pause by hand.

    width/height carry the ego2 render size purely to reserve the right box:
    a src-less <video> is 300x150 by default, so without them every clip
    jumps from ~150 px to its real height the moment it attaches, shoving the
    plots the reviewer is reading down the page. A clip of some other size
    still renders correctly (both pages set object-fit:contain), it just
    letterboxes inside the reserved box."""
    attrs = ['controls', 'muted', 'loop', 'playsinline', 'preload="none"',
             f'width="{int(width)}"', f'height="{int(height)}"',
             f'data-src="{html.escape(src, quote=True)}"']
    if cls:
        attrs.append(f'class="{html.escape(cls, quote=True)}"')
    if style:
        attrs.append(f'style="{html.escape(style, quote=True)}"')
    return f"<video {' '.join(attrs)}></video>"
