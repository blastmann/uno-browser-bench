const MAX_TITLE = 160;
const MAX_VALUE = 96;

export function redactText(value, maxLength = MAX_VALUE) {
  if (value === undefined || value === null) return "";
  return String(value)
    .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, "<email>")
    .replace(/\+?\d[\d ()-]{7,}\d/g, "<phone>")
    .replace(/\b\d{4,}\b/g, "<num>")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxLength);
}

export function redactUrl(value) {
  try {
    const url = new URL(String(value));
    if (!["http:", "https:"].includes(url.protocol)) return "<non-http>";
    const safePath = url.pathname
      .split("/")
      .filter(Boolean)
      .slice(0, 3)
      .map((part) => redactText(part, 48))
      .join("/");
    return `${url.protocol}//${url.hostname}${safePath ? `/${safePath}` : "/"}`;
  } catch {
    return "<invalid-url>";
  }
}

export function redactElement(element) {
  return {
    role: redactText(element?.role, 32),
    id: redactText(element?.id, 48),
    text: redactText(element?.text, MAX_TITLE),
  };
}

export function redactObservation(observation) {
  return {
    url: redactUrl(observation?.url),
    title: redactText(observation?.title, MAX_TITLE),
    elements: Array.isArray(observation?.elements) ? observation.elements.slice(0, 32).map(redactElement) : [],
    events: Array.isArray(observation?.events)
      ? observation.events.slice(-64).map((event, index) => ({
          t: Number.isFinite(event?.t) ? event.t : index,
          action: redactText(event?.action, 32),
          target: redactText(event?.target, 64),
          ...(event?.value ? { value: redactText(event.value) } : {}),
        }))
      : [],
  };
}

export function redactHistoryItem(item) {
  return {
    url: redactUrl(item?.url),
    title: redactText(item?.title, MAX_TITLE),
    lastVisitTime: Number.isFinite(item?.lastVisitTime) ? item.lastVisitTime : undefined,
    visitCount: Number.isFinite(item?.visitCount) ? Math.min(item.visitCount, 1000) : undefined,
  };
}
