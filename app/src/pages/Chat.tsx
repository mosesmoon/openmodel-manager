import { useI18n } from "../i18n";
import ChatView from "./ChatView";

export default function Chat() {
  const { t } = useI18n();
  return (
    <>
      <h1>{t("nav.chat")}</h1>
      <ChatView mode="chat" />
    </>
  );
}
