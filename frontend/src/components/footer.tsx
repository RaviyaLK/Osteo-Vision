import { FiGithub, FiLinkedin } from "react-icons/fi";

const Footer = () => {
  return (
    <footer className="bg-blue-800 text-white py-4">
      <div className="container py-4 mx-auto flex justify-between items-center">
        <div className=" text-lg">
          <p> Osteo-Vision | © 2025 All Rights Reserved </p>
        </div>
        <div className="flex space-x-4">
          <a
            href="https://github.com/RaviyaLK"
            target="_blank"
            rel="noopener noreferrer"
            className="text-white hover:text-gray-400"
          >
            <FiGithub size={24} />
          </a>
          <a
            href="https://www.linkedin.com/in/ravindu-yasintha-wijesekara"
            target="_blank"
            rel="noopener noreferrer"
            className="text-white hover:text-gray-400 "
          >
            <FiLinkedin size={24} />
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
