import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { MetaProvider } from "./state";
import "./styles.css";

const host = document.getElementById("root");
if (!host) throw new Error("#root is missing from index.html");

createRoot(host).render(
  <StrictMode>
    <BrowserRouter>
      <MetaProvider>
        <App />
      </MetaProvider>
    </BrowserRouter>
  </StrictMode>,
);
