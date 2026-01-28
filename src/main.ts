import { digitize } from "./digitize";

declare global {
  interface Window {
    digitize: typeof digitize;
  }
}

window.digitize = digitize;
