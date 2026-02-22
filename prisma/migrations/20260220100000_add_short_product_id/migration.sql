-- AlterTable
ALTER TABLE "shorts" ADD COLUMN "product_id" TEXT;

-- CreateIndex
CREATE UNIQUE INDEX "shorts_product_id_key" ON "shorts"("product_id");
