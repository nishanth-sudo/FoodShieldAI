import { CONFIG } from "../lib/config";

describe("Frontend Configuration and Validation Rules", () => {
  test("should have correct API URL configured", () => {
    expect(CONFIG.API_URL).toBeDefined();
    expect(typeof CONFIG.API_URL).toBe("string");
  });

  test("should enforce file size limits under 10MB", () => {
    // Limit must be exactly 5MB based on requirements
    expect(CONFIG.MAX_FILE_SIZE).toBe(5 * 1024 * 1024);
  });

  test("should specify accepted image extensions", () => {
    expect(CONFIG.ACCEPTED_FILE_EXTENSIONS).toContain(".png");
    expect(CONFIG.ACCEPTED_FILE_EXTENSIONS).toContain(".jpg");
    expect(CONFIG.ACCEPTED_FILE_EXTENSIONS).toContain(".webp");
  });

  test("should check confidence threshold for warning triggers", () => {
    expect(CONFIG.CONFIDENCE_THRESHOLD).toBe(0.70);
  });
});
