
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface ReportGenerationDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  isGenerating: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ReportGenerationDialog({
  open,
  onOpenChange,
  isGenerating,
  onConfirm,
  onCancel,
}: ReportGenerationDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        {isGenerating ? (
          <div className="flex flex-col items-center justify-center p-6">
            <VisuallyHidden>
              <DialogTitle>Generating Report</DialogTitle>
            </VisuallyHidden>
            <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-blue-500 mb-4"></div>
            <p className="text-lg font-semibold">Generating your report...</p>
          </div>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>Report Generation Complete</DialogTitle>
              <DialogDescription>
                Your medical report is ready to be downloaded.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={onCancel}>
                Cancel
              </Button>
              <Button variant="outline" onClick={onConfirm}>
                Download Report
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}