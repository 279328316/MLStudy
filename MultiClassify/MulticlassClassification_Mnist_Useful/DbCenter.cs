
using Jc.Core;
using Jc.Core.Data;
using Jc.Core.Data.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MulticlassClassification_Mnist_Useful
{
    /// <summary>
    /// DataBaseCenter
    /// </summary>
    public class Dbc
    {   //CodeCreatorDb
        public static DbContext Db = DbContext.CreateDbContext("Nice", DatabaseType.MsSql);
    }
}
