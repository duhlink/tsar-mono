import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    projects: [
      'packages/agent/vitest.config.ts',
      'packages/ai/vitest.config.ts',
      'packages/coding-agent/vitest.config.ts',
    ],
  },
});
