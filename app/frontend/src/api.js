import axios from "axios";

const API_URL = "http://localhost:8000";

export async function fetchSearch(query) {
  const res = await axios.get(`${API_URL}/search`, { params: { query } });
  return res.data;
}
