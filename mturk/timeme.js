// TIME ME CODE
// TimeMe.js
(function () { var e, t; e = this, t = function () { var r = { startStopTimes: {}, idleTimeoutMs: 3e4, currentIdleTimeMs: 0, checkStateRateMs: 250, active: !1, idle: !1, currentPageName: "default-page-name", timeElapsedCallbacks: [], userLeftCallbacks: [], userReturnCallbacks: [], trackTimeOnElement: function (e) { var t = document.getElementById(e); t && (t.addEventListener("mouseover", function () { r.startTimer(e) }), t.addEventListener("mousemove", function () { r.startTimer(e) }), t.addEventListener("mouseleave", function () { r.stopTimer(e) }), t.addEventListener("keypress", function () { r.startTimer(e) }), t.addEventListener("focus", function () { r.startTimer(e) })) }, getTimeOnElementInSeconds: function (e) { var t = r.getTimeOnPageInSeconds(e); return t || 0 }, startTimer: function (e, t) { if (e || (e = r.currentPageName), void 0 === r.startStopTimes[e]) r.startStopTimes[e] = []; else { var n = r.startStopTimes[e], i = n[n.length - 1]; if (void 0 !== i && void 0 === i.stopTime) return } r.startStopTimes[e].push({ startTime: t || new Date, stopTime: void 0 }), r.active = !0, r.idle = !1 }, stopAllTimers: function () { for (var e = Object.keys(r.startStopTimes), t = 0; t < e.length; t++)r.stopTimer(e[t]) }, stopTimer: function (e, t) { e || (e = r.currentPageName); var n = r.startStopTimes[e]; void 0 !== n && 0 !== n.length && (void 0 === n[n.length - 1].stopTime && (n[n.length - 1].stopTime = t || new Date), r.active = !1) }, getTimeOnCurrentPageInSeconds: function () { return r.getTimeOnPageInSeconds(r.currentPageName) }, getTimeOnPageInSeconds: function (e) { var t = r.getTimeOnPageInMilliseconds(e); return void 0 === t ? void 0 : t / 1e3 }, getTimeOnCurrentPageInMilliseconds: function () { return r.getTimeOnPageInMilliseconds(r.currentPageName) }, getTimeOnPageInMilliseconds: function (e) { var t = r.startStopTimes[e]; if (void 0 !== t) { for (var n = 0, i = 0; i < t.length; i++) { var s = t[i].startTime, o = t[i].stopTime; void 0 === o && (o = new Date), n += o - s } return Number(n) } }, getTimeOnAllPagesInSeconds: function () { for (var e = [], t = Object.keys(r.startStopTimes), n = 0; n < t.length; n++) { var i = t[n], s = r.getTimeOnPageInSeconds(i); e.push({ pageName: i, timeOnPage: s }) } return e }, setIdleDurationInSeconds: function (e) { var t = parseFloat(e); if (!1 !== isNaN(t)) throw { name: "InvalidDurationException", message: "An invalid duration time (" + e + ") was provided." }; return r.idleTimeoutMs = 1e3 * e, this }, setCurrentPageName: function (e) { return r.currentPageName = e, this }, resetRecordedPageTime: function (e) { delete r.startStopTimes[e] }, resetAllRecordedPageTimes: function () { for (var e = Object.keys(r.startStopTimes), t = 0; t < e.length; t++)r.resetRecordedPageTime(e[t]) }, resetIdleCountdown: function () { r.idle && r.triggerUserHasReturned(), r.idle = !1, r.currentIdleTimeMs = 0 }, callWhenUserLeaves: function (e, t) { this.userLeftCallbacks.push({ callback: e, numberOfTimesToInvoke: t }) }, callWhenUserReturns: function (e, t) { this.userReturnCallbacks.push({ callback: e, numberOfTimesToInvoke: t }) }, triggerUserHasReturned: function () { if (!r.active) for (var e = 0; e < this.userReturnCallbacks.length; e++) { var t = this.userReturnCallbacks[e], n = t.numberOfTimesToInvoke; (isNaN(n) || void 0 === n || 0 < n) && (t.numberOfTimesToInvoke -= 1, t.callback()) } r.startTimer() }, triggerUserHasLeftPage: function () { if (r.active) for (var e = 0; e < this.userLeftCallbacks.length; e++) { var t = this.userLeftCallbacks[e], n = t.numberOfTimesToInvoke; (isNaN(n) || void 0 === n || 0 < n) && (t.numberOfTimesToInvoke -= 1, t.callback()) } r.stopAllTimers() }, callAfterTimeElapsedInSeconds: function (e, t) { r.timeElapsedCallbacks.push({ timeInSeconds: e, callback: t, pending: !0 }) }, checkState: function () { for (var e = 0; e < r.timeElapsedCallbacks.length; e++)r.timeElapsedCallbacks[e].pending && r.getTimeOnCurrentPageInSeconds() > r.timeElapsedCallbacks[e].timeInSeconds && (r.timeElapsedCallbacks[e].callback(), r.timeElapsedCallbacks[e].pending = !1); !1 === r.idle && r.currentIdleTimeMs > r.idleTimeoutMs ? (r.idle = !0, r.triggerUserHasLeftPage()) : r.currentIdleTimeMs += r.checkStateRateMs }, visibilityChangeEventName: void 0, hiddenPropName: void 0, listenForVisibilityEvents: function () { void 0 !== document.hidden ? (r.hiddenPropName = "hidden", r.visibilityChangeEventName = "visibilitychange") : void 0 !== document.mozHidden ? (r.hiddenPropName = "mozHidden", r.visibilityChangeEventName = "mozvisibilitychange") : void 0 !== document.msHidden ? (r.hiddenPropName = "msHidden", r.visibilityChangeEventName = "msvisibilitychange") : void 0 !== document.webkitHidden && (r.hiddenPropName = "webkitHidden", r.visibilityChangeEventName = "webkitvisibilitychange"), document.addEventListener(r.visibilityChangeEventName, function () { document[r.hiddenPropName] ? r.triggerUserHasLeftPage() : r.triggerUserHasReturned() }, !1), window.addEventListener("blur", function () { r.triggerUserHasLeftPage() }), window.addEventListener("focus", function () { r.triggerUserHasReturned() }), document.addEventListener("mousemove", function () { r.resetIdleCountdown() }), document.addEventListener("keyup", function () { r.resetIdleCountdown() }), document.addEventListener("touchstart", function () { r.resetIdleCountdown() }), window.addEventListener("scroll", function () { r.resetIdleCountdown() }), setInterval(function () { r.checkState() }, r.checkStateRateMs) }, websocket: void 0, websocketHost: void 0, setUpWebsocket: function (e) { if (window.WebSocket && e) { var t = e.websocketHost; try { r.websocket = new WebSocket(t), window.onbeforeunload = function () { r.sendCurrentTime(e.appId) }, r.websocket.onopen = function () { r.sendInitWsRequest(e.appId) }, r.websocket.onerror = function (e) { console && console.log("Error occurred in websocket connection: " + e) }, r.websocket.onmessage = function (e) { console && console.log(e.data) } } catch (e) { console && console.error("Failed to connect to websocket host.  Error:" + e) } } return this }, websocketSend: function (e) { r.websocket.send(JSON.stringify(e)) }, sendCurrentTime: function (e) { var t = { type: "INSERT_TIME", appId: e, timeOnPageMs: r.getTimeOnCurrentPageInMilliseconds(), pageName: r.currentPageName }; r.websocketSend(t) }, sendInitWsRequest: function (e) { var t = { type: "INIT", appId: e }; r.websocketSend(t) }, initialize: function (e) { var t = r.idleTimeoutMs || 30, n = r.currentPageName || "default-page-name", i = void 0, s = void 0; e && (t = e.idleTimeoutInSeconds || t, n = e.currentPageName || n, i = e.websocketOptions, s = e.initialStartTime), r.setIdleDurationInSeconds(t).setCurrentPageName(n).setUpWebsocket(i).listenForVisibilityEvents(), r.startTimer(void 0, s) } }; return r }, "undefined" != typeof module && module.exports ? module.exports = t() : "function" == typeof define && define.amd ? define([], function () { return e.TimeMe = t() }) : e.TimeMe = t() }).call(this);

TimeMe.initialize({
    currentPageName: "task",
    idleTimeoutInSeconds: 30
});

$(document).ready(function() {
    $('#submitButton').click(function () {
        try {
            $('input[name=tm]').attr('value', TimeMe.getTimeOnCurrentPageInSeconds());
        } catch {
        }
        return true;
    });
});
