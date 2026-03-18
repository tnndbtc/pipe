const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir:  './playwright',
  timeout:  30000,
  retries:  0,
  workers:  1,
  use: { baseURL: 'http://localhost:19999', headless: true },
  reporter: 'list',
});
