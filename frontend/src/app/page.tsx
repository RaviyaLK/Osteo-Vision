
import UploadForm from "@/components/UploadForm";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
export default function Home() {
  return (
    <main>
       <ToastContainer />
      <UploadForm />
    </main>
  );
}

