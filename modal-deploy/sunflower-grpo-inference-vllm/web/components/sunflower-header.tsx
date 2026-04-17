import Image from "next/image";
import logoIcon from "@/assets/logo_icon.png";
import { ThemeToggle } from "./theme-toggle";

export function SunflowerHeader() {
  return (
    <header className="flex items-center gap-3">
      <Image
        src={logoIcon}
        alt="Sunflower"
        width={36}
        height={36}
        priority
        className="h-9 w-9 shrink-0"
      />
      <div className="text-lg font-semibold tracking-tight">Sunbird AI</div>
      <div className="ml-auto flex items-center gap-2">
        <span className="inline-flex h-6 items-center rounded-full border border-border bg-card px-2.5 text-xs text-muted-foreground">
          Sunflower GRPO · Test
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
}
