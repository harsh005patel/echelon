export default async function handler(req, res) {
  const backendBase = process.env.BACKEND_URL;
  if (!backendBase) {
    return res.status(500).json({ error: "BACKEND_URL binding is not defined" });
  }

  const targetUrl = new URL('/health', backendBase);

  try {
    const backendRes = await fetch(targetUrl);
    const data = await backendRes.json();
    return res.status(backendRes.status).json(data);
  } catch (err) {
    return res.status(502).json({ error: "Failed to connect to backend service", details: err.message });
  }
}
