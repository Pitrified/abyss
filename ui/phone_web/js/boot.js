// A counter on the status line while the page loads, before any module runs.
// app.js imports three.js and tasks-vision statically, so nothing of it runs
// until they have arrived; over a slow link that alone takes seconds. This is
// a classic script, loaded first and on its own, so it starts at once.
//
// app.js names the phase it is in on window.abyssLoading and sets it to null
// when frames start; the counter then stops and leaves the line to app.js.

window.abyssLoading = "modules";
const bootTimer = setInterval(() => {
  const line = document.getElementById("status");
  if (window.abyssLoading === null) {
    clearInterval(bootTimer);
    return;
  }
  if (line) line.textContent = `loading ${window.abyssLoading}: ${(performance.now() / 1000).toFixed(0)} s`;
}, 250);
