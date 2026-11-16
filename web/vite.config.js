import { createReadStream, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// Serves ../fixtures at /fixtures/ so the bench page can load local weights.
export default defineConfig({
  server: { watch: null },
  plugins: [
    {
      name: "fixtures",
      configureServer(server) {
        server.middlewares.use("/fixtures/", (req, res, next) => {
          // WebLLM asks for Hugging Face style paths: <model>/resolve/main/<file>.
          const path = decodeURIComponent((req.url ?? "").split("?")[0]).replace(
            "/resolve/main/",
            "/",
          );
          if (path.includes("..")) return next();
          const file = fileURLToPath(new URL(`../fixtures${path}`, import.meta.url));
          try {
            res.setHeader("Content-Length", statSync(file).size);
            createReadStream(file).pipe(res);
          } catch {
            res.statusCode = 404;
            res.end();
          }
        });
      },
    },
  ],
});
